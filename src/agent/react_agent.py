"""Agente ReAct usando LangChain + Azure OpenAI."""

from __future__ import annotations

import logging
import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

REACT_PROMPT = PromptTemplate.from_template(
    """Você é um assistente que ajuda analistas a entender e justificar recomendações de ofertas bancárias.

Use as ferramentas disponíveis para responder. Se a pergunta puder ser respondida diretamente, responda — não invente uso de tool.

Ferramentas:
{tools}

Formato obrigatório de resposta:

Question: a pergunta do usuário
Thought: raciocínio sobre o próximo passo
Action: nome_da_ferramenta (uma de [{tool_names}])
Action Input: input para a ferramenta
Observation: resultado retornado pela ferramenta
... (repita Thought/Action/Observation quantas vezes for necessário)
Thought: Sei a resposta final
Final Answer: resposta consolidada para o usuário, em português

Question: {input}
{agent_scratchpad}"""
)


def _default_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
    )


def build_react_agent(
    tools: list[Tool],
    llm: BaseLanguageModel | None = None,
    *,
    max_iterations: int = 8,
    verbose: bool = False,
) -> AgentExecutor:
    """Monta o AgentExecutor ReAct com as tools fornecidas."""
    if len(tools) < 3:
        logger.warning(
            "Agente recebeu %d tools — o esperado é pelo menos 3", len(tools)
        )

    chat = llm or _default_llm()
    agent = create_react_agent(llm=chat, tools=tools, prompt=REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
