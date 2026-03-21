"""
Script to split dataset_chunks.md into separate .md files based on policy_id (document name).
"""

import re
import os


def split_dataset_by_policy(input_file: str, output_dir: str = "split_policies"):
    """
    Splits the dataset_chunks.md file into separate .md files based on policy/document name.
    
    Args:
        input_file: Path to the dataset_chunks.md file
        output_dir: Directory to save the split files (default: 'split_policies')
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match document headers (## document_name.pdf)
    # This captures the document name and its content until the next document header
    pattern = r'^## (.+?\.pdf)\s*\n'
    
    # Find all document headers and their positions
    matches = list(re.finditer(pattern, content, re.MULTILINE))
    
    if not matches:
        print("No policy documents found in the file.")
        return
    
    print(f"Found {len(matches)} policy documents:")
    
    # Extract header section (before first document)
    header_section = content[:matches[0].start()]
    
    saved_files = []
    
    for i, match in enumerate(matches):
        document_name = match.group(1)
        
        # Get the content for this document
        start_pos = match.start()
        
        # End position is either the start of next document or end of file
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        document_content = content[start_pos:end_pos]
        
        # Create a safe filename (remove special characters)
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', document_name)
        safe_filename = safe_filename.replace('.pdf', '')
        output_filename = f"{safe_filename}.md"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write the document content to a separate file
        # Include modified header section with document-specific info
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write a header for this specific policy
            f.write(f"# Policy Chunks: {document_name}\n\n")
            f.write("---\n\n")
            f.write(document_content)
        
        saved_files.append(output_filename)
        print(f"  {i+1}. {document_name} -> {output_filename}")
    
    print(f"\nSuccessfully split into {len(saved_files)} files in '{output_dir}/' directory.")
    return saved_files


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file path
    input_file = os.path.join(script_dir, "dataset_chunks.md")
    
    # Output directory path
    output_dir = os.path.join(script_dir, "split_policies")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Split the dataset
    split_dataset_by_policy(input_file, output_dir)


if __name__ == "__main__":
    main()
