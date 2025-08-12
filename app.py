import operator
from typing import Annotated, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# =========================
# 初期設定
# =========================
st.set_page_config(page_title="要件定義 生成エージェント", layout="wide")
st.title("🧩 要件定義 生成エージェント（Streamlit版）")
st.caption("LangChain + LangGraph で、ペルソナ→インタビュー→評価→要件定義を自動生成")

# .env 読み込み（OPENAI_API_KEY 等）
load_dotenv()


# =========================
# データモデル
# =========================
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景")


class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="ペルソナのリスト")


class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")


class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )


class EvaluationResult(BaseModel):
    reason: str = Field(..., description="判断の理由")
    is_sufficient: bool = Field(..., description="情報が十分かどうか")


class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(default=0, description="反復回数")
    is_information_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )
    # 評価理由（UIで表示したいのでステートにも載せておく）
    evaluation_reason: str = Field(default="", description="評価理由")


# =========================
# コンポーネント
# =========================
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーインタビュー用の多様なペルソナを作成する専門家です。",
                ),
                (
                    "human",
                    f"以下のユーザーリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "各ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})


class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        questions = self._generate_questions(user_request=user_request, personas=personas)
        answers = self._generate_answers(personas=personas, questions=questions)
        interviews = self._create_interviews(personas=personas, questions=questions, answers=answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザー要件に基づいて適切な質問を生成する専門家です。",
                ),
                (
                    "human",
                    "以下のペルソナに関連するユーザーリクエストについて、1つの質問を生成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n"
                    "ペルソナ: {persona_name} - {persona_background}\n\n"
                    "質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください。",
                ),
            ]
        )
        question_chain = question_prompt | self.llm | StrOutputParser()

        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        return question_chain.batch(question_queries)

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたは以下のペルソナとして回答しています: {persona_name} - {persona_background}"),
                ("human", "質問: {question}"),
            ]
        )
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        answer_queries = [
            {"persona_name": persona.name, "persona_background": persona.background, "question": question}
            for persona, question in zip(personas, questions)
        ]
        return answer_chain.batch(answer_queries)

    def _create_interviews(self, personas: list[Persona], questions: list[str], answers: list[str]) -> list[Interview]:
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]


class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。",
                ),
                (
                    "human",
                    "以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まったかどうかを判断してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "インタビュー結果:\n{interview_results}",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )


class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたは収集した情報に基づいて要件文書を作成する専門家です。"),
                (
                    "human",
                    "以下のユーザーリクエストと複数のペルソナからのインタビュー結果に基づいて、要件文書を作成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "インタビュー結果:\n{interview_results}\n"
                    "要件文書には以下のセクションを含めてください:\n"
                    "1. プロジェクト概要\n"
                    "2. 主要機能\n"
                    "3. 非機能要件\n"
                    "4. 制約条件\n"
                    "5. ターゲットユーザー\n"
                    "6. 優先順位\n"
                    "7. リスクと軽減策\n\n"
                    "出力は必ず日本語でお願いします。\n\n要件文書:",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )


class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k or 5)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewState)
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)
        workflow.set_entry_point("generate_personas")
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # 直近 k 人分に対して質問・回答（UIの k と揃える）
        k = self.persona_generator.k
        target = state.personas[-k:] if k > 0 else state.personas
        new_interviews: InterviewResult = self.interview_conductor.run(state.user_request, target)
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_generator.run(
            state.user_request, state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run_full(self, user_request: str) -> dict[str, Any]:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return {
            "requirements_doc": final_state["requirements_doc"],
            "personas": final_state["personas"],
            "interviews": final_state["interviews"],
            "evaluation_reason": final_state.get("evaluation_reason", ""),
            "iterations": final_state["iteration"],
            "is_information_sufficient": final_state["is_information_sufficient"],
        }


# =========================
# UI（Streamlit）
# =========================
with st.sidebar:
    st.header("⚙️ 設定")
    model = st.selectbox(
        "OpenAI モデル",
        ["gpt-4o", "gpt-4.1", "gpt-4o-mini"],
        index=0,
    )
    temperature = st.slider("temperature", 0.0, 1.0, 0.0, 0.1)
    k = st.number_input("生成するペルソナ人数 k", min_value=1, max_value=10, value=5, step=1)
    st.caption("※ 環境変数 `OPENAI_API_KEY` か `.env` に API キーを設定してください。")

st.subheader("📝 ユーザーリクエスト")
default_task = "スマートフォン向けの健康管理アプリを開発したい"
task = st.text_area("作成したいアプリケーションについて記載してください", value=default_task, height=120)

run = st.button("🚀 生成を実行", type="primary", disabled=not bool(task.strip()))

if run:
    try:
        with st.spinner("モデル初期化中..."):
            llm = ChatOpenAI(model=model, temperature=temperature)

        agent = DocumentationAgent(llm=llm, k=int(k))

        with st.spinner("ペルソナ作成～インタビュー～評価～要件定義を生成中..."):
            result = agent.run_full(user_request=task)

        # ========== 出力表示 ==========
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 👥 生成されたペルソナ")
            for idx, p in enumerate(result["personas"], start=1):
                with st.expander(f"Persona #{idx}: {p.name}", expanded=False):
                    st.write(p.background)

        with col2:
            st.markdown("### 🗣️ インタビュー（質問と回答）")
            for idx, iv in enumerate(result["interviews"], start=1):
                with st.expander(f"Interview #{idx} - {iv.persona.name}", expanded=False):
                    st.markdown(f"**質問**: {iv.question}")
                    st.markdown(f"**回答**: {iv.answer}")

        st.markdown("### 🧪 情報十分性評価")
        st.write("十分性:", "✅ 十分" if result["is_information_sufficient"] else "⚠️ 不十分（最大5回で打ち切り）")
        if result["evaluation_reason"]:
            st.info(result["evaluation_reason"])
        st.caption(f"反復回数: {result['iterations']}")

        st.markdown("### 📄 生成された要件定義書")
        st.markdown(result["requirements_doc"])

        st.download_button(
            "⬇️ 要件定義書をMarkdownでダウンロード",
            data=result["requirements_doc"].encode("utf-8"),
            file_name="requirements.md",
            mime="text/markdown",
        )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.exception(e)
