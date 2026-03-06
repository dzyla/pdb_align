class AlignmentFailedError(Exception):
    """Raised when the alignment fails."""
    pass

class ChainNotFoundError(Exception):
    """Raised when a specified chain is not found in the structure."""
    pass

class ParsingError(Exception):
    """Raised when there is an error parsing a structure file."""
    pass
