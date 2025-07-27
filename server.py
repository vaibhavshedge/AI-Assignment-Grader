from fastapi import FastAPI, Request, HTTPException, Depends
import uvicorn
import openai
import os
import asyncio
import ollama
import sys
from pydantic import BaseModel,Field
from typing import Dict, Any, Optional, Union, List
import requests
from functools import lru_cache
import logging
from dotenv import load_dotenv
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class APISettings(BaseModel):
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    search_engine_id: Optional[str] = Field(default=None)

@lru_cache()
def get_settings():
    settings = APISettings(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        search_engine_id=os.getenv('SEARCH_ENGINE_ID')
    )

    # Log configuration status (but don't expose actual keys)
    logger.info(f"OPENAI_API_KEY set: {'✅' if settings.openai_api_key else '❌'}")
    logger.info(f"GOOGLE_API_KEY set: {'✅' if settings.google_api_key else '❌'}")
    logger.info(f"SEARCH_ENGINE_ID set: {'✅' if settings.search_engine_id else '❌'}")
    logger.info(f"Python version: {sys.version}")


    return settings


class ParseFileRequest(APISettings):
    file_path: str
    file_type: str

class PlagiarismRequest(APISettings):
    text: str
    similarity_threshold: Optional[int] = 40

class GradeRequest(APISettings):
    text: str
    rubric: str
    model: Optional[str] = "gpt-4.1"

class ErrorResponse(BaseModel):
    detail: str

class GradeResponse(BaseModel):
    grade: str

class PlagiarismResult(BaseModel):
    url: str
    similarity: int

class PlagiarismResponse(BaseModel):
    results: List[PlagiarismResult]


class ApiCheckRequest(BaseModel):
    openai_api_key: str
    google_api_key: str
    search_engine_id: str

class ApiCheckResponse(BaseModel):
    openai_api_key: str
    google_api_key: str



app = FastAPI(
    title="Assignment Grader API",
    description="API for parsing, grading, and checking plagiarism in academic assignments",
    version="1.0.0",
    responses={
        500: {"model": ErrorResponse}
    }
)


@app.get("/")
async def home():
    return {"message": "Assignment Grader API", "status": "running", "version": "1.0.0"}

def get_api_keys(request, settings):
    """Get API keys from request or environment"""
    openai_key = getattr(request, "openai_api_key", None) or settings.openai_api_key
    google_key = getattr(request, "google_api_key", None) or settings.google_api_key
    search_id = getattr(request, "search_engine_id", None) or settings.search_engine_id
    
    return {
        "openai_api_key": openai_key,
        "google_api_key": google_key,
        "search_engine_id": search_id
    }

async def parse_pdf(file_path: str) -> str:
    try:
        import fitz  
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    
    except ImportError:
        raise HTTPException(status_code=500, detail="PyMuPDF not installed. Install with 'pip install pymupdf'")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing PDF: {str(e)}")


async def parse_docx(file_path: str) -> str:
    try:
        from docx import Document  # Import only when needed
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx not installed. Install with 'pip install python-docx'")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing DOCX: {str(e)}")


@app.post("/tools/parse_file", response_model=str)
async def parse_file(request: ParseFileRequest):
    try:
        file_path = request.file_path
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        ext = request.file_type
        if ext == ".pdf":
            return await parse_pdf(file_path)
        elif ext == ".docx":
            return await parse_docx(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")


@app.post("/tools/check_plagiarism", response_model=PlagiarismResponse)
async def check_plagiarism(request: PlagiarismRequest, settings: APISettings = Depends(get_settings)):
    try:
        # Get API keys
        keys = get_api_keys(request, settings)
        
        if not keys["google_api_key"] or not keys["search_engine_id"]:
            raise HTTPException(status_code=500, detail="Google API key or Search Engine ID not configured")
            
        from fuzzywuzzy import fuzz  # Import only when needed
        
        text = request.text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        # Take first 300 chars for the search query
        query = text[:300].replace("\n", " ").strip()
        
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": keys["google_api_key"],
            "cx": keys["search_engine_id"]
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                            detail=f"Google API error: {response.text}")
                            
        data = response.json()
        results = data.get("items", [])
        
        plagiarism_results = [
            PlagiarismResult(
                url=item["link"],
                similarity=fuzz.token_set_ratio(text, item.get("snippet", ""))
            )
            for item in results
        ]
        
        # Sort by similarity (highest first)
        plagiarism_results.sort(key=lambda x: x.similarity, reverse=True)
        
        # Filter by threshold if provided
        threshold = request.similarity_threshold or 0
        if threshold > 0:
            plagiarism_results = [i for i in plagiarism_results if i.similarity >= threshold]
        
        return PlagiarismResponse(results=plagiarism_results)
    except ImportError:
        raise HTTPException(status_code=500, detail="fuzzywuzzy not installed. Install with 'pip install fuzzywuzzy python-Levenshtein'")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking plagiarism: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking plagiarism: {str(e)}")


async def call_openai_api(prompt: str, api_key: str, model: str = "tinyllama") -> str:
    try:
        response = await asyncio.to_thread(
            ollama.chat,
            model="tinyllama",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.5,
                "num_predict": 1024,
            },
        )
        return response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

@app.post("/tools/grade_text", response_model=GradeResponse)
async def grade_text(request: GradeRequest, settings: APISettings = Depends(get_settings)):
    try:
        text = request.text
        rubric = request.rubric
        model = request.model or "gpt-3.5-turbo"
        
        # Get API keys
        keys = get_api_keys(request, settings)
        
        if not text.strip() or not rubric.strip():
            raise HTTPException(status_code=400, detail="Text and rubric cannot be empty")
        
        if not keys["openai_api_key"]:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        prompt = f"""You are an academic grader. Grade the following assignment based on the rubric. 
                Respond with only the grade (Example A,B,C ...):

                Rubric: {rubric}

                Assignment: {text}"""
        
        grade = await call_openai_api(prompt, keys["openai_api_key"], model)
        return GradeResponse(grade=grade)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error grading text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error grading text: {str(e)}")


@app.post("/tools/generate_feedback", response_model=str)
async def generate_feedback(request: GradeRequest, settings: APISettings = Depends(get_settings)):
    try:
        text = request.text
        rubric = request.rubric
        model = request.model or "gpt-3.5-turbo"
        
        # Get API keys
        keys = get_api_keys(request, settings)
        
        if not text.strip() or not rubric.strip():
            raise HTTPException(status_code=400, detail="Text and rubric cannot be empty")
        
        if not keys["openai_api_key"]:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        prompt = f"""You are a teacher. Give constructive feedback to a student based on this rubric and assignment.

                Rubric: {rubric}

                Assignment: {text}

                Write your feedback below:"""
        
        feedback = await call_openai_api(prompt, keys["openai_api_key"], model)
        return feedback
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")

@app.post("/debug/check_keys")
async def check_keys(request: ApiCheckRequest, settings: APISettings = Depends(get_settings)):

    keys = get_api_keys(request, settings)

    client = openai.OpenAI(api_key=keys['openai_api_key'])
    response = client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]
    )

    reply = response.choices[0].message.content

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
    "key": keys['google_api_key'] ,
    "cx": keys['search_engine_id'],
    "q": "MCP Code example for LangChain-mcp-adapter"
    }

    response = requests.get(url, params=params).json()

    return ApiCheckResponse(
        openai_api_key= "✅API is Up and working"if len(reply) > 0 else "❌ API is not working Use different API Key",
        google_api_key= "✅API is Up and working"if len(response.get("items", [])) > 0 else "❌ API is not working Use different API Key"
    )




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)