from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split the input text into chunks using RecursiveCharacterTextSplitter.
    :param text: The text to be split.
    :param chunk_size: Maximum size of each chunk.
    :param chunk_overlap: Overlap size between consecutive chunks.
    :return: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
