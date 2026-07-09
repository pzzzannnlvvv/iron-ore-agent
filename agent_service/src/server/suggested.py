from typing import List
import json

from loguru import logger
from fastapi import APIRouter, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from .schemas import SuggestedRequest, SuggestedResponse


router = APIRouter(prefix="/agent/api", tags=["suggested"])


class QuestionList(BaseModel):
    """Question list model for parsing LLM output"""
    questions: List[str] = Field(..., description="Recommended question list, maximum three")


# Create output parser
output_parser = PydanticOutputParser(pydantic_object=QuestionList)


def get_system_prompt():
    """Get system prompt with format instructions"""
    format_instructions = output_parser.get_format_instructions()
    # Escape curly braces in format_instructions to prevent ChatPromptTemplate from treating them as variables
    escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    return f"""You are a question recommendation assistant. Based on the user's input prompt, generate up to three relevant, thoughtful questions.

Requirements:
1. Questions should be specific and targeted
2. Questions should have depth and thinking value
3. Each question should be a complete sentence ending with a question mark
4. Generate at most three questions. If the user input is already specific enough, you can generate fewer than three.

Please output strictly in the following JSON format:
{escaped_format_instructions}"""


def create_suggestion_chain():
    """Create question suggestion chain with multi-message format"""
    # Get LLM type from agent mapping
    llm_type = AGENT_LLM_MAP.get("suggested", "block")
    llm = get_llm_by_type(llm_type)

    # Create prompt template with multi-message format
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt()),
        ("user", "{prompt}"),
    ])

    # Build chain: prompt -> llm -> parser
    chain = prompt_template | llm | output_parser
    return chain


@router.post("/suggested")
async def suggest_questions(request: SuggestedRequest) -> SuggestedResponse:
    """
    Recommend related questions based on user input prompt

    Args:
        request: Request containing prompt field

    Returns:
        SuggestedResponse: Response containing recommended question list
    """
    try:
        logger.info(f"Received question recommendation request, prompt: {request.prompt}")

        # Create chain
        chain = create_suggestion_chain()

        # Prepare input
        chain_input = {
            "prompt": request.prompt,
        }

        # Invoke chain
        result: QuestionList = await chain.ainvoke(chain_input)

        # Ensure no more than 3 questions
        questions = result.questions[:3] if result.questions else []

        logger.info(f"Generated recommended questions: {questions}")

        return SuggestedResponse(questions=questions)

    except Exception as e:
        logger.error(f"Question recommendation API error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommended questions: {str(e)}")