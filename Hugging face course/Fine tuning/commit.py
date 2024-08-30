from huggingface_hub import Repository
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

def push_model_and_tokenizer(
    local_repo_path: str,
    hub_repo_url: str,
    commit_message: str,
    model_dir: Path,
    tokenizer_dir: Path
):
    """
    Push a model and tokenizer to a Hugging Face Hub repository.

    Args:
        local_repo_path (str): Path to the local folder where the repository will be stored.
        hub_repo_url (str): URL of the Hugging Face Hub repository.
        commit_message (str): Commit message for the update.
        model_dir (Path): Path to the local directory containing the model files.
        tokenizer_dir (Path): Path to the local directory containing the tokenizer files.
    
    Raises:
        ValueError: If the model or tokenizer directories are invalid.
        RuntimeError: If pushing to the repository fails.
    """
    local_repo_path = Path(local_repo_path)
    
    # Check if the repository already exists locally
    if local_repo_path.exists() and local_repo_path.is_dir():
        print(f"Using existing repository at {local_repo_path}.")
        repo = Repository(local_repo_path)
    else:
        print(f"Cloning repository from {hub_repo_url} to {local_repo_path}.")
        repo = Repository(local_repo_path, clone_from=hub_repo_url)

    # Load the model and tokenizer
    if not model_dir.is_dir() or not tokenizer_dir.is_dir():
        raise ValueError("The provided model or tokenizer directories are invalid.")

    try:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    # Pull the latest changes
    try:
        repo.git_pull()
    except Exception as e:
        raise RuntimeError(f"Failed to pull the latest changes: {e}")

    # Save the model and tokenizer to the repository
    try:
        model.save_pretrained(repo.local_dir)
        tokenizer.save_pretrained(repo.local_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to save model or tokenizer: {e}")

    # Commit and push the changes
    try:
        repo.git_add()
        repo.git_commit(commit_message)
        repo.push_to_hub()
    except Exception as e:
        raise RuntimeError(f"Failed to commit or push changes: {e}")

    print("Model and tokenizer successfully pushed to the Hub.")
