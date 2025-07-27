import streamlit as st
import requests
import json
import os
import tempfile
import humanize 
import logging
from typing import Union
from datetime import datetime
from fpdf import FPDF
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SERVER_URL = "http://localhost:8088"
API_TIMEOUT = 200

# App configuration
st.set_page_config(page_title="Assignment Grader", page_icon="📝", layout="wide")
st.title("📝 Assignment Grader")
st.markdown("Upload assignments and grade them automatically with AI")

# Initialize session state
def init_session_state():
    """Initialize session state variables if they don't exist"""
    if 'server_url' not in st.session_state:
        st.session_state['server_url'] = DEFAULT_SERVER_URL
    
    # Load API keys from environment
    st.session_state['openai_api_key'] = os.getenv('OPENAI_API_KEY')
    st.session_state['google_api_key'] = os.getenv('GOOGLE_API_KEY')
    st.session_state['search_engine_id'] = os.getenv('SEARCH_ENGINE_ID')

init_session_state()

def mask_api_key(key: str):
    """Mask API key for logging purposes"""
    if not key:
        return "[Empty]"  
    if len(key) > 10:
        return f"🔑 {key[:4]}...{key[-4:]}"
    # For very short keys, mask everything
    return "🔑 [Hidden]"

def call_api_tool(tool_name: str, data: dict) -> Union[dict, None]:

    url = f"{st.session_state['server_url']}/tools/{tool_name}"

    # Prepare request data with API keys
    request_data = data.copy()
    request_data.update({
        "openai_api_key": st.session_state['openai_api_key'],
        "google_api_key": st.session_state['google_api_key'],
        "search_engine_id": st.session_state['search_engine_id']
    })

    # Prepare masked data for logging
    log_data = request_data.copy()
    log_data['openai_api_key'] = mask_api_key(log_data['openai_api_key'])
    log_data['google_api_key'] = mask_api_key(log_data['google_api_key'])
    
    logger.info(f"Calling {tool_name} with data: {json.dumps(log_data)}")

    try:
        response = requests.post(
            url, 
            json=request_data,
            headers={"Content-Type": "application/json"}, 
            timeout=API_TIMEOUT
        )

        if response.status_code != 200:
            error_message = f"Error {response.status_code} from server: {response.text}"
            logger.error(error_message)
            st.error(error_message)
            return None
        
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    except Exception as e:
        error_message = f"Error connecting to server: {str(e)}"
        logger.error(error_message)
        st.error(error_message)
        return None

def check_server_connection():
    """Check if the server is online and return status"""
    try:
        response = requests.get(f"{st.session_state['server_url']}/", timeout=API_TIMEOUT)
        if response.status_code == 200:
            st.success("✅ Server is online!")
            st.json(response.json())
        else:
            st.warning(f"⚠️ Server responded with status {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"❌ Failed to connect:")
        with st.expander("Check error detail"):
            st.error(f"{str(e)}")


# Sidebar configuration
with st.sidebar:
    st.header("Server Configuration")

    with st.expander("Server Settings", expanded=True):
        server_url = st.text_input("API Server URL", value=st.session_state['server_url'])
    
        if st.button("Save Server URL"):
            st.session_state['server_url'] = server_url
            st.success(f"✅ Server URL updated to {server_url}")

    st.write("---")
    
    st.subheader("Server Status")
    if st.button("Check Server Connection"):
        check_server_connection()


    st.write("---")
    st.subheader("Test API Keys")
    
    key_col1, key_col2 = st.columns([3, 1])

    status_styles = {
        "active": {"emoji": "🔑", "color": "#28a745", "text": "Active"},
        "missing": {"emoji": "🔒", "color": "#dc3545", "text": "Missing"}
    }

    keys = {
        "OpenAI API": st.session_state.get('openai_api_key'),
        "Google API": st.session_state.get('google_api_key'),
        "Search Engine ID": st.session_state.get('search_engine_id')
    }

    with key_col1:
        for key_name, key_value in keys.items():
            status = "active" if key_value else "missing"
            st.caption(f"{status_styles[status]['emoji']} {key_name}")

    with key_col2:
        for key_name, key_value in keys.items():
            status = "active" if key_value else "missing"
            st.markdown(
                f"<div style='color: {status_styles[status]['color']}; font-weight: bold;'>" # <div style='color:red; font-weight:bold'>Active</div>
                f"{status_styles[status]['text']}"
                f"</div>",
                unsafe_allow_html=True
            )
    
    if st.button("Test API Keys"):
        try:
            # Test endpoint
            data = {
                "openai_api_key": st.session_state.get('openai_api_key'),
                "google_api_key": st.session_state.get('google_api_key'),
                "search_engine_id": st.session_state.get('search_engine_id'),
            }
            
            response = requests.post(
                f"{st.session_state['server_url']}/debug/check_keys", 
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                st.success("✅ API keys test successful!")
                with st.expander("Test Results"):
                    st.json(response.json())
            else:
                st.error(f"❌ API keys test failed: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")





# Create tabs
tab1, tab2, tab3 = st.tabs(["📄 Upload", "✏️ Grade", "📊 Results"])

with tab1:
    st.header("Upload Assignment")
    
    # Show instructions
    with st.expander("Upload Instructions", expanded=False):
        st.markdown("""
        1. Upload a PDF or Word document containing the assignment
        2. Click 'Process Document' to extract text
        3. Review the document preview before grading
        """)
    
    uploaded_file = st.file_uploader("Choose a file", 
                                    type=["pdf", "docx"], 
                                    help="Upload PDF or Word files only")

    if uploaded_file is not None:
        # Get file info
        file_size = len(uploaded_file.getvalue())
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        file_icon = "📄" if file_extension == ".pdf" else "📝"
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"{file_icon} Uploaded: {uploaded_file.name}")
        with col2:
            st.info(f"Size: {humanize.naturalsize(file_size)}")
        
        # Create progress bar for file processing
        progress_bar = st.progress(0)
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
            progress_bar.progress(50)
            
        # Store file info in session state
        st.session_state["file_path"] = file_path
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["file_type"] = file_extension
        progress_bar.progress(100)
        
        process_button = st.button(
                "🔍 Process Document", 
                help="Extract text from the document",
                use_container_width=True
            )
                
        if process_button:
            try:
                with st.spinner("Extracting text from document..."):
                    # Call API to process document
                    result = call_api_tool("parse_file", {
                        "file_path": file_path,
                        "file_type": file_extension
                    })
                    
                    if result is None:
                        st.error("❌ Failed to process document. Check server connection.")
                    elif isinstance(result, str):
                        # Get word count and character count
                        words = result.split()
                        word_count = len(words)
                        char_count = len(result)
                        
                        # Store text in session state
                        st.session_state["document_text"] = result
                        st.session_state["word_count"] = word_count
                        st.session_state["char_count"] = char_count
                        
                        # Show success message with extraction stats
                        st.success(f"✅ Document processed successfully!")
                        
                        # Display document statistics
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Words", f"{word_count:,}")
                        with stat_col2:
                            st.metric("Characters", f"{char_count:,}")
                        with stat_col3:
                            reading_time = max(1, round(word_count / 200))  # Avg reading speed
                            st.metric("Reading Time", f"{reading_time} min")
                        
                        # Long document warning
                        if word_count > 5000:
                            st.warning(
                                f"⚠️ Long document detected ({word_count:,} words). "
                                "Processing might take longer."
                            )
                        
                        # Show document preview
                        with st.expander("Document Preview", expanded=True):
                            preview = result[:1000] + ("..." if len(result) > 1000 else "") 
                            st.text_area(
                                "Document Content",
                                value=preview,
                                height=300,
                                disabled=True
                            )
                            
                    else:
                        # Handle structured result
                        st.session_state["document_text"] = str(result)
                        st.success(f"✅ Document processed!")
                        
                        # Show structured data preview
                        with st.expander("Document Data", expanded=True):
                            st.json(result)
                            
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.info("Please try uploading the file again or contact support.")
    else:
        # Show placeholder when no file is uploaded
        st.info("👆 Please upload a document to begin")




# Tab 2: Grade Assignment
with tab2:
    st.header("Grading Configuration")

    # Check if document is loaded
    if "document_text" not in st.session_state:
        st.warning("⚠️ Please upload and process a document first.")
    st.success(
            f"✅ Document loaded: {st.session_state.get('file_name', 'Unknown')}"
        )
    
    st.subheader("Grading Rubric")

    # Default rubric templates
    rubric_templates = {
    "Default Academic": 
    """Content (40%): Demonstrates comprehensive understanding of the topic with relevant, accurate information.
Structure (20%): Presents a logical organization with clear introduction, coherent body paragraphs, and insightful conclusion.
Analysis (30%): Offers critical analysis supported by compelling evidence and thoughtful reasoning.
Grammar & Style (10%): Exhibits polished writing with proper grammar, appropriate academic tone, and consistent formatting.
    """,
    "Technical Report": 
    """Accuracy (35%): Presents precise technical information with clear explanations and appropriate terminology.
Methodology (25%): Details a systematic approach with well-justified methods appropriate to the investigation.
Results (25%): Communicates findings clearly using effective data visualizations and concise explanations.
Conclusions (15%): Derives logical conclusions directly supported by the data analysis and contextualizes the significance.
    """,
    "Creative Writing": 
    """Originality (30%): Exhibits innovative ideas, unique perspective, and creative approach to the subject matter.
Structure (20%): Employs effective narrative structure with purposeful pacing and meaningful organization.
Character/Scene Development (30%): Creates vivid, multi-dimensional characters and/or richly detailed settings.
Language & Style (20%): Utilizes expressive language with varied sentence structure and distinctive voice.
    """,
}   
    
    template_choice = st.selectbox(
        "Select a template or create your own:",
        ["Default Academic", "Technical Report", "Creative Writing", "Custom"],
    )

    rubric_text = rubric_templates.get(template_choice,"") if template_choice != 'Custom' else ""

    rubric = st.text_area(
        label="Enter your Grading Rubric here",
        height=200,
        value=rubric_text,
        help="Specify the criteria on which the assignment should be graded"
    )


    # Plagiarism check and grading options
    col1, col2 = st.columns(2)
    with col1:
        check_plagiarism = st.checkbox("Check for plagiarism", value=True)

        if check_plagiarism:
            similarity_threshold = st.slider(
                "Similarity threshold (%)",
                min_value=1,
                max_value=90,
                value=50,
                help="Minimum similarity percentage to flag potential plagiarism",
            )

    with col2:
        grade_model = st.selectbox(
            "AI Model for Grading",
            ["gpt-4.1","gpt-4o-mini","gpt-3.5-turbo", "gpt-4"],
            help="Select the AI model to use for grading (affects accuracy and cost)",
        )
    

    # Grade Assignment button
    if "document_text" in st.session_state:
        if st.button("Grade Assignment", type="primary",use_container_width=True):
            # Store rubric in session
            st.session_state["rubric"] = rubric

            with st.spinner("Grading in progress..."):
                progress_bar = st.progress(0)

                # Optional plagiarism check
                if check_plagiarism:
                    st.info("📊 Checking for plagiarism...")

                    plagiarism_data = {
                        "text": st.session_state["document_text"],
                        "similarity_threshold": similarity_threshold
                    }

                    plagiarism_results = call_api_tool(
                        "check_plagiarism", plagiarism_data
                    )
                    st.session_state["plagiarism_results"] = plagiarism_results

                    progress_bar.progress(33)
                else:
                    progress_bar.progress(33)
                
                # Generate grade
                st.info("🧮 Generating grade...")

                grade_data = {
                    "text": st.session_state["document_text"],
                    "rubric": rubric,
                    "model": grade_model
                }

                grade_results = call_api_tool("grade_text", grade_data)
                st.session_state["grade_results"] = grade_results

                progress_bar.progress(66)

                # Generate feedback
                st.info("✍️ Generating detailed feedback...")

                feedback_data = {
                    "text": st.session_state["document_text"],
                    "rubric": rubric,
                    "model":grade_model
                }

                feedback = call_api_tool("generate_feedback", feedback_data)
                st.session_state["feedback"] = feedback

                progress_bar.progress(100)

                if grade_results is not None or feedback is not None:
                    st.success("✅ Grading completed!")
                    st.balloons()
                else:
                    st.error(
                        "❌ Grading process encountered errors. Please check your server connection and API settings."
                    )


# Tab 3: Results
with tab3:
    st.header("Grading Results")
    if 'file_name' in st.session_state:
        st.subheader(f"Results for: {st.session_state['file_name']}")

    # st.code(st.session_state['grade_results'])
    # st.code(st.session_state["plagiarism_results"])

    # Create columns for grade  and feedback display
    col1, col2 = st.columns([1, 3])

    with col1:
        grade = "Not available"
        if 'grade_results' in st.session_state and st.session_state['grade_results'] is not None:  # st.session_state['grade_results'] = {'grade': 'A'}
            if isinstance(st.session_state['grade_results'], dict):
                grade = st.session_state['grade_results'].get('grade', 'Not available')
                st.markdown(f"## Grade: {grade}")
        else:
            st.warning("Grade information is not available.")
            st.metric("Grade", "Not available")
    
    with col2:
        feedback = "Feedback is not available."
        if 'feedback' in st.session_state and st.session_state['feedback'] is not None:
            st.subheader("Feedback")
            feedback = st.session_state['feedback']
            st.markdown(feedback)
        else:
            st.warning("Feedback is not available.")
    

    # Display plagiarism results if available
    plagiarism_data = []
    if 'plagiarism_results' in st.session_state and st.session_state['plagiarism_results']:
        st.subheader("Plagiarism Check")
        results = st.session_state['plagiarism_results'] # {'results': [{'url': str, 'similarity': float}, {}, {}, ...]}


        if isinstance(results, dict) and 'results' in results and isinstance(results['results'], list):
            if len(results['results']) > 0:
                st.markdown("**Similarity matches found:**")

                for item in results['results']:
                    url = item.get('url', '')
                    similarity = item.get('similarity', 0)

                    plagiarism_data.append({
                            "url": url,
                            "similarity": similarity
                        })

                    if similarity > 70:
                        st.warning(f"⚠️ High similarity ({similarity}%): [{url}]({url})")
                    elif similarity > 40:
                        st.info(f"ℹ️ Moderate similarity ({similarity}%): [{url}]({url})")
                    else:
                        st.success(f"✅ Low similarity ({similarity}%): [{url}]({url})")
        
        else:
            st.json(results)  # Display raw results if format is unknown
        

    st.write("---")
    st.subheader("Download Report")

    def generate_text_report():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = st.session_state.get('file_name', 'Document')
            
            report = f"""# Grading Report

## Assignment Information
- **Filename:** {filename}
- **Generated:** {timestamp}

## Grade
{grade}

## Feedback
{feedback}

"""
            if plagiarism_data:
                report += "## Plagiarism Check Results\n\n"
                for item in plagiarism_data:
                    similarity = item.get('similarity', 0)
                    level = "HIGH" if similarity > 70 else "MODERATE" if similarity > 40 else "LOW"
                    report += f"- **{level}** similarity ({similarity}%): {item.get('url', '')}\n"
                
            return report
    

    report_text = generate_text_report()
    filename = f"grading_report_{datetime.now().strftime('%Y%m%d')}.md"
    
    st.download_button(
        label="Download Report",
        data=report_text,
        file_name=filename,
        mime="text/markdown",
        key="download_text_report"
    )
    
 




    

                        