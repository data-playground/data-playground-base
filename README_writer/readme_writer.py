# %%
# Setup
import ast
import os

from google import genai
from google.cloud import secretmanager


def skeletonize_code(file_path):
    """Extracts only the structure, imports, and docstrings from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        tree = ast.parse(source)
        skeleton = []

        # 1. Grab Imports (for Tool Detection)
        imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            skeleton.append(f"Imports: {', '.join(set(imports))}")

        # 2. Grab Module Docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            skeleton.append(f"Module Description: {module_doc}")

        # 3. Extract Classes and Functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                doc = ast.get_docstring(node) or "No documentation provided."
                # Extracting arguments for functions
                args = ""
                if isinstance(node, ast.FunctionDef):
                    args = ", ".join(arg.arg for arg in node.args.args)
                
                skeleton.append(f"{'Class' if isinstance(node, ast.ClassDef) else 'Function'}: {name}({args})")
                skeleton.append(f"   Docstring: {doc}")

        return "\n".join(skeleton)
    except Exception as e:
        return f"Error parsing {file_path}: {e}"
    
import json


def get_json_schema(file_path):
    """Returns the structure/keys of a JSON file instead of the full data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # If it's a list, look at the first item
    sample = data[0] if isinstance(data, list) and len(data) > 0 else data
    
    if isinstance(sample, dict):
        return f"JSON Structure: Keys are {list(sample.keys())}"
    return "JSON contains a simple list or primitive data."

IGNORE_DIRS = {'.git', '.github', '__pycache__', 'venv', 'env'}

def analyze_folder(folder_path):
    """Gathers context specifically for a single process folder."""
    context = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.py', '.json', '.yaml', '.yml', '.sql')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        context.append(f"--- FILE: {file} ---\n{content[:3000]}")
                except:
                    continue
    return "\n".join(context)

def new_gather_context(folder):
    process_context = []
    for root, _, files in os.walk(folder):
        if root.startswith(os.path.join(folder, 'venv')) or root == os.path.join(folder, '__pycache__'):
            continue
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('.py'):
                process_context.append(f"--- SCRIPT: {file} ---\n{skeletonize_code(full_path)}")
            elif file.endswith('.json'):
                process_context.append(f"--- DATA: {file} ---\n{get_json_schema(full_path)}")

    return process_context

def get_key(SECRET_NAME):
    """
        Get API Key from Google Secret Manager
    """
    # Initialize the Secret Manager client
    SMclient = secretmanager.SecretManagerServiceClient()

    # Set the project ID 
    project_id = "impactful-post-292301"

    # Build the request to access the secret version
    request = {"name": f"projects/{project_id}/secrets/{SECRET_NAME}/versions/latest"}

    # Access the secret version
    response = SMclient.access_secret_version(request)

    # Get the secret value
    SECRET_VALUE = response.payload.data.decode("UTF-8")

    return SECRET_VALUE

def generate_readme(folder):
    client = genai.Client(api_key=get_key('Gemini-API'))
    if os.path.isdir(folder) and folder not in IGNORE_DIRS:
        print(f"📂 Analyzing process: {folder} ...")
        
        context = new_gather_context(folder)
        if not context: 
            return

        # Updated Prompt with Portfolio-specific sections
        prompt = f"""
        You are an expert Technical Portfolio Curator. Create a high-quality README.md 
        for the project in the folder: '{folder}'.

        ### INPUT CONTEXT:
        {context}

        ### INSTRUCTIONS:
        1. **Title**: Create a punchy, descriptive title.
        2. **Difficulty Level**: Analyze the code complexity (use of classes, decorators, 
            concurrency, or advanced math) and label it as: [Beginner, Intermediate, or Advanced].
        3. **Tools Used**: Extract the tech stack (e.g., Pandas, Scikit-learn, JSON, 
            Requests). Present these as a clean list or a set of Markdown badges.
        4. **Data Interaction**: Explicitly describe how the Python script uses the 
            accompanying JSON data (e.g., "The script parses 'config.json' to dynamically 
            route API calls").
        5. **Key Features**: Use bullet points for the main functionalities.
        6. **How to Run**: Provide the exact terminal commands.

        ### OUTPUT FORMAT:
        - Use a professional, clean Markdown layout.
        - Include a 'Portfolio Context' section at the top.
        - DO NOT use code blocks for the entire response; just the raw Markdown content.
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt, 
            )
            readme_path = os.path.join(folder, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                # Cleaning response to ensure it's pure Markdown
                clean_content = response.text.replace("```markdown", "").replace("```", "").strip()
                f.write(clean_content)
            print(f"✅ Created README for {folder}")
        except Exception as e:
            print(f"❌ Failed {folder}: {e}")

# if __name__ == "__main__":
#     generate_readme()
# %%
