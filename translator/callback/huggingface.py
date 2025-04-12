from translator.callback.base import BaseCallback
from translator.parser.dynamic import DynamicDataParser
from huggingface_hub import HfApi, create_repo, whoami
import os
import yaml


class HuggingFaceCallback(BaseCallback):
    def on_finish_save_translated(self, instance: DynamicDataParser):
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

        # Initialize the HF API
        api = HfApi(token=hf_token)

        # Validate token before proceeding
        try:
            user_info = whoami(token=hf_token)
            hf_username = user_info.get("name")

            # Debug info
            print(f"Debug - User info: {user_info}")

            # Check if token has write permission
            # The role field can be either a string or a list depending on the API version
            token_role = user_info.get("auth", {}).get("accessToken", {}).get("role")
            has_write_access = False

            if isinstance(token_role, list):
                has_write_access = "write" in token_role
            elif isinstance(token_role, str):
                has_write_access = token_role == "write"

            if not has_write_access:
                print("❌ Token does not have write permission")
                print("⚠️ Your token can only read repositories but not write to them.")
                print(
                    "⚠️ Generate a new token with write access at: https://huggingface.co/settings/tokens"
                )
                return

            print(f"✅ Authenticated as: {hf_username} with write permissions")
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            print("⚠️ Please check your token has the correct permissions.")
            print(
                "⚠️ Generate a new token with write access at: https://huggingface.co/settings/tokens"
            )
            return

        # Get the parser name and target language
        parser_name = instance.parser_name

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
        repo_name = f"{original_dataset_repo_name}-vi"
        full_repo_name = f"{hf_username}/{repo_name}"

        print(f"🔄 Preparing to upload translated dataset to {full_repo_name}")

        # Get paths
        output_dir = instance.output_dir
        translated_file_path = os.path.join(output_dir, f"{parser_name}_vi.json")

        # Validate that translated file exists
        if not os.path.exists(translated_file_path):
            print(f"⚠️ Translated file not found at {translated_file_path}")
            return

        # Create README content and path
        readme_path = os.path.join(output_dir, "README.md")

        try:
            # Create README content
            readme_content = self._prepare_readme(
                original_dataset_repo_name, hf_username, repo_name
            )
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)

            # Create or get the repository on Hugging Face Hub
            try:
                # Check if repository exists
                api.repo_info(repo_id=full_repo_name, repo_type="dataset")
                print(f"📦 Repository {full_repo_name} already exists, will update")
            except Exception:
                # Create repository if it doesn't exist
                print(f"🆕 Creating new repository {full_repo_name}")
                try:
                    create_repo(
                        repo_id=full_repo_name,
                        token=hf_token,
                        repo_type="dataset",
                        private=False,
                        exist_ok=True,
                    )
                except Exception as repo_error:
                    print(f"❌ Error creating repository: {str(repo_error)}")
                    print("⚠️ Please check your token permissions or try again later.")
                    raise

            # Upload the dataset file to Hugging Face Hub
            print(f"⬆️ Uploading dataset to {full_repo_name}")
            api.upload_file(
                path_or_fileobj=translated_file_path,
                path_in_repo="data.json",
                repo_id=full_repo_name,
                repo_type="dataset",
                commit_message="Upload translated dataset",
            )

            # Upload README file
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=full_repo_name,
                repo_type="dataset",
                commit_message="Add README.md",
            )

            # Add metadata directly in the README.md file (removing separate metadata handling)
            print(f"✅ Successfully uploaded translated dataset to {full_repo_name}")
            print(f"🔗 Visit: https://huggingface.co/datasets/{full_repo_name}")

        except Exception as e:
            print(f"❌ Error uploading dataset to Hugging Face: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            # Clean up temporary README if it was created
            if os.path.exists(readme_path):
                try:
                    os.remove(readme_path)
                except Exception:
                    pass

    def _prepare_readme(self, original_dataset_name, huggingface_username, repo_name):
        """
        Prepare README content based on the template with proper metadata.

        Args:
            instance: The parser instance
            original_dataset_name: Name of the original dataset

        Returns:
            README content with metadata
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
            template_content = f.read()

        # Create metadata in the same structure as before
        metadata = {
            "language": "vi",
            "license": "mit",
            "tags": [
                "translated",
                "machine-translated",
                "multilingual",
                "vi",
            ],
        }

        metadata_yaml = yaml.dump(metadata, sort_keys=False)
        metadata_header = f"---\n{metadata_yaml}---\n\n"

        # Replace placeholders in template
        template_content = template_content.replace("[DATASET_NAME]", repo_name)
        template_content = template_content.replace(
            "[YOUR_HF_USERNAME]", huggingface_username
        )
        original_dataset_link = self.get_hf_dataset_link(original_dataset_name)
        template_content = template_content.replace(
            "[ORIGINAL_DATASET_REPO_LINK]", original_dataset_link
        )
        # Combine metadata header with the template content
        return metadata_header + template_content

    def get_hf_dataset_link(self, dataset_name: str) -> str:
        """
        Generate the Hugging Face dataset link for the given dataset name.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            str: The Hugging Face dataset link in markdown.
        """
        return f"[{dataset_name}](https://huggingface.co/datasets/{dataset_name})"
