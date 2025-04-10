import re
from typing import Optional

class LinkedInPostOutputParser:
    """Parser for extracting LinkedIn post content from LLM output."""
    
    @staticmethod
    def _remove_markdown(text: str) -> str:
        """Remove markdown formatting from text while preserving LinkedIn-friendly elements.
        
        Args:
            text: The text containing markdown formatting
            
        Returns:
            Clean text with markdown formatting removed but preserving structure
        """
        # Remove bold/italic markers but keep content
        text = re.sub(r'\*\*(.+?)\*\*', lambda m: m.group(1).upper(), text)
        text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)|_(.+?)_', 
                     lambda m: m.group(1) or m.group(2), text)
        
        # Remove links but keep text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        # Preserve hashtags
        text = re.sub(r'(?<!\w)#(\w+)', r'#\1', text)
        
        # Remove backticks
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        # Convert bullet points to LinkedIn-friendly format
        text = re.sub(r'^[-*+]\s+', '• ', text, flags=re.MULTILINE)
        
        # Convert numbered lists to plain text
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def parse(text: str) -> Optional[str]:
        """Extract content between [START_POST] and [END_POST] markers and remove markdown.
        
        Args:
            text: The text containing the LinkedIn post content
            
        Returns:
            The extracted post content with markdown removed, or None if markers not found
        """
        start_marker = "[START_POST]"
        end_marker = "[END_POST]"
        
        try:
            start_idx = text.index(start_marker) + len(start_marker)
            end_idx = text.index(end_marker)
            content = text[start_idx:end_idx].strip()
            return LinkedInPostOutputParser._remove_markdown(content)
        except ValueError:
            return None