"""
Custom AdalFlow Components.

This module provides custom, robust components that override or extend the
default behavior of the AdalFlow library. It is designed to handle specific
challenges encountered during experiments, such as malformed LLM outputs,
by replacing brittle components with more resilient implementations.
"""

import re
import logging
from adalflow.optim.text_grad.tgd_optimizer import CustomizedXMLParser, TGDData, TGDOptimizer

# Configure logger for this module
log = logging.getLogger(__name__)


class RobustXMLParser(CustomizedXMLParser):
    """
    An overridden version of the default XML parser that uses regex.

    This parser is designed to be resilient against malformed XML output from
    the Teacher LLM. Instead of using a strict XML engine, it uses regular
    expressions to find and extract content within specific tags, gracefully
    ignoring any surrounding malformed structures or extraneous text.

    It inherits from the original `CustomizedXMLParser` to ensure it remains a
    valid drop-in replacement within the AdalFlow framework.
    """
    def call(self, input: str) -> TGDData:
        """
        Parses the XML-like response from the optimizer LLM robustly.

        Args:
            input (str): The raw string output from the Teacher LLM.

        Returns:
            TGDData: A structured data object containing the parsed fields.
        """
        # Helper function to safely extract content from a specific tag.
        def extract_tag_content(tag_name: str, text: str) -> str:
            """
            Finds a tag and returns its content using a non-greedy regex search.
            
            Args:
                tag_name (str): The name of the XML tag (e.g., "reasoning").
                text (str): The string to search within.

            Returns:
                str: The extracted content, or an empty string if not found.
            """
            # The pattern looks for <tag_name>...content...</tag_name>
            # re.DOTALL allows '.' to match newlines for multi-line content.
            pattern = f"<{tag_name}>(.*?)</{tag_name}>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # group(1) contains the text *inside* the tags.
                return match.group(1).strip()
            return ""

        try:
            clean_input = input.strip()
            
            # Use the robust helper for each expected field
            reasoning = extract_tag_content("reasoning", clean_input)
            method = extract_tag_content("method", clean_input)
            proposed_variable = extract_tag_content("proposed_variable", clean_input)
            
            # Log a warning if the most critical field is missing, for debugging.
            if not proposed_variable:
                log.warning(
                    f"Could not find a valid <proposed_variable> tag in the "
                    f"optimizer's response. Full output: {clean_input}"
                )

            return TGDData(
                reasoning=reasoning,
                method=method,
                proposed_variable=proposed_variable
            )
            
        except Exception as e:
            # Catch any other unexpected errors during the process
            log.error(f"A critical error occurred in RobustXMLParser.call: {e}")
            return TGDData(
                reasoning=f"Critical parsing error: {e}", 
                method="Error",
                proposed_variable=input  # Return original input on critical failure
            )


class CustomTGDOptimizer(TGDOptimizer):
    """
    An overridden version of the TGDOptimizer.
    
    This class inherits all the functionality of the original `TGDOptimizer` but
    makes one critical change: it replaces the default, brittle XML parser with
    our custom `RobustXMLParser`. This ensures the optimization loop does not
    crash due to poorly formatted LLM responses.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the optimizer, then overrides the default parser.

        Args:
            *args: Positional arguments to pass to the parent `TGDOptimizer`.
            **kwargs: Keyword arguments to pass to the parent `TGDOptimizer`.
        """
        # First, call the parent's __init__ to perform all standard setup.
        super().__init__(*args, **kwargs)
        
        # Now, override the default parser with our robust implementation.
        self.output_parser = RobustXMLParser()
        
        # It's also crucial to update the system prompt's template variables
        # to use the output format string from our new parser.
        self.optimizer_system_prompt.prompt_kwargs["output_format_str"] = self.output_parser.get_output_format_str()