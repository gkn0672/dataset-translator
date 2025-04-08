from translator.callback.base import BaseCallback
from huggingface_hub import HfApi, create_repo
import os
import json


class HuggingFaceCallback(BaseCallback):
    def on_finish_save_translated(self, instance):
        """
        Called when the parser has finished saving the translated data.
        Uploads the translated dataset to Hugging Face Hub.
        """
        print(
            f"Parser {instance.parser_name} has finished saving the translated data, preparing to upload to Hugging Face Hub..."
        )

        # Get HF authentication token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "⚠️ No Hugging Face token found in environment. Please set HF_TOKEN environment variable."
            )
            return

        # Get HF username from environment
        hf_username = os.environ.get("HF_USERNAME")
        if not hf_username:
            print(
                "⚠️ No Hugging Face username found in environment. Please set HF_USERNAME environment variable."
            )
            return

        # Initialize the HF API
        api = HfApi(token=hf_token)

        # Get the parser name and target language
        parser_name = instance.parser_name
        target_lang = instance.target_lang

        # Determine original dataset name from instance
        original_dataset = (
            instance.dataset_name if hasattr(instance, "dataset_name") else parser_name
        )

        # Extract the dataset name without namespace if it has one
        if "/" in original_dataset:
            original_dataset_repo_name = original_dataset.split("/")[-1]
        else:
            original_dataset_repo_name = original_dataset

        # Create repository name
        repo_name = f"{original_dataset_repo_name}-{target_lang}"
        full_repo_name = f"{hf_username}/{repo_name}"

        print(f"🔄 Preparing to upload translated dataset to {full_repo_name}")

        # Get paths
        output_dir = instance.output_dir
        translated_file_path = os.path.join(
            output_dir, f"{parser_name}_translated_{target_lang}.json"
        )

        # Validate that translated file exists
        if not os.path.exists(translated_file_path):
            print(f"⚠️ Translated file not found at {translated_file_path}")
            return

        try:
            # Create or get the repository on Hugging Face Hub
            try:
                # Check if repository exists
                api.repo_info(repo_id=full_repo_name, repo_type="dataset")
                print(f"📦 Repository {full_repo_name} already exists, will update")
            except Exception:
                # Create repository if it doesn't exist
                print(f"🆕 Creating new repository {full_repo_name}")
                create_repo(
                    repo_id=full_repo_name,
                    token=hf_token,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True,
                )

            # Create README content
            readme_content = self._prepare_readme(original_dataset_repo_name)
            readme_path = os.path.join(output_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)

            # Upload the dataset file to Hugging Face Hub
            print(f"⬆️ Uploading dataset to {full_repo_name}")
            api.upload_file(
                path_or_fileobj=translated_file_path,
                path_in_repo="data.json",
                repo_id=full_repo_name,
                repo_type="dataset",
                commit_message=f"Upload translated dataset ({target_lang})",
            )

            # Upload README file
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=full_repo_name,
                repo_type="dataset",
                commit_message="Add README",
            )

            # Add metadata (including tags)
            metadata = {
                "language": target_lang,
                "license": "mit",
                "original_dataset": original_dataset,
                "tags": [
                    "translated",
                    "machine-translated",
                    "multilingual",
                    target_lang,
                ],
            }

            # Update the dataset card
            api.update_repo_card(
                repo_id=full_repo_name, repo_type="dataset", metadata=metadata
            )

            print(f"✅ Successfully uploaded translated dataset to {full_repo_name}")
            print(f"🔗 Visit: https://huggingface.co/datasets/{full_repo_name}")

        except Exception as e:
            print(f"❌ Error uploading dataset to Hugging Face: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            # Clean up temporary README if it was created
            if os.path.exists(readme_path):
                os.remove(readme_path)

    def _prepare_readme(self, original_dataset_name):
        """
        Prepare README content based on the template.

        Args:
            original_dataset_name: Name of the original dataset

        Returns:
            README content
        """
        # Get the template path
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "callback",
            "utils",
            "template.md",
        )

        # Read the template
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # Replace placeholders
        content = template.replace("[DATASET_NAME]", original_dataset_name)
        content = content.replace("[ORIGINAL_DATASET_REPO_NAME]", original_dataset_name)

        return content
