"""PDF rendering module for converting documents to PDF format."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
from pygments.formatters import HtmlFormatter
from tqdm import tqdm
from weasyprint import HTML

from dataset_factory.storage.dataset import Document


# CSS styling for professional PDF appearance
PDF_CSS = """
@page {
    size: Letter;
    margin: 1in 0.75in;
    
    @top-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}

.content {
    margin-top: 0;
}

h1 {
    font-size: 20pt;
    color: #2c3e50;
    margin-top: 25px;
    margin-bottom: 15px;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 8px;
}

h2 {
    font-size: 16pt;
    color: #34495e;
    margin-top: 20px;
    margin-bottom: 12px;
}

h3 {
    font-size: 13pt;
    color: #34495e;
    margin-top: 15px;
    margin-bottom: 10px;
}

h4, h5, h6 {
    font-size: 11pt;
    color: #34495e;
    margin-top: 12px;
    margin-bottom: 8px;
}

p {
    margin: 10px 0;
    text-align: justify;
}

ul, ol {
    margin: 10px 0;
    padding-left: 30px;
}

li {
    margin: 5px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 10pt;
}

th {
    background-color: #34495e;
    color: white;
    padding: 10px;
    text-align: left;
    font-weight: bold;
}

td {
    padding: 8px 10px;
    border: 1px solid #ddd;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 10pt;
    color: #c7254e;
}

pre {
    background-color: #f8f9fa;
    border: 1px solid #ddd;
    border-left: 3px solid #3498db;
    padding: 12px;
    overflow-x: auto;
    margin: 15px 0;
    border-radius: 4px;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: inherit;
    font-size: 9pt;
}

blockquote {
    border-left: 4px solid #3498db;
    padding-left: 15px;
    margin: 15px 0;
    color: #555;
    font-style: italic;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

hr {
    border: none;
    border-top: 1px solid #bdc3c7;
    margin: 20px 0;
}

/* Syntax highlighting styles */
""" + HtmlFormatter(style='colorful').get_style_defs('.codehilite')


def _create_html_document(document: Document) -> str:
    """
    Create complete HTML document from a Document object.
    
    Args:
        document: Document to convert to HTML
        
    Returns:
        Complete HTML string with styling
    """
    # Convert markdown to HTML with extensions
    md = markdown.Markdown(
        extensions=[
            TableExtension(),
            FencedCodeExtension(),
            CodeHiliteExtension(linenums=False, guess_lang=False),
            TocExtension(),
            'nl2br',
            'sane_lists',
        ]
    )
    
    content_html = md.convert(document.content)
    
    # Build complete HTML document (content only, no metadata)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{document.id}</title>
        <style>
            {PDF_CSS}
        </style>
    </head>
    <body>
        <div class="content">
            {content_html}
        </div>
    </body>
    </html>
    """
    
    return html


def render_document_to_pdf(document: Document, output_path: Path) -> None:
    """
    Render a single document to PDF.
    
    Args:
        document: Document to render
        output_path: Path where PDF should be saved
        
    Raises:
        Exception: If rendering fails
    """
    try:
        # Create HTML
        html_content = _create_html_document(document)
        
        # Convert to PDF
        HTML(string=html_content).write_pdf(output_path)
        
    except Exception as e:
        raise Exception(f"Failed to render document {document.id}: {e}") from e


def _render_single_document(args: tuple[Document, Path]) -> tuple[bool, str, str | None]:
    """
    Helper function for parallel processing.
    
    Args:
        args: Tuple of (document, output_path)
        
    Returns:
        Tuple of (success, document_id, error_message)
    """
    document, output_path = args
    try:
        render_document_to_pdf(document, output_path)
        return (True, document.id, None)
    except Exception as e:
        return (False, document.id, str(e))


def render_documents_to_pdfs(
    documents: list[Document],
    output_dir: Path,
    parallel: bool = True,
    workers: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Render multiple documents to PDFs.
    
    Args:
        documents: List of documents to render
        output_dir: Directory where PDFs should be saved
        parallel: Whether to use parallel processing
        workers: Number of parallel workers (default: CPU count / 2)
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with rendering statistics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if workers is None:
        workers = max(1, os.cpu_count() // 2)
    
    # Prepare arguments for rendering
    render_args = []
    for document in documents:
        # Sanitize filename
        safe_filename = f"{document.id}.pdf"
        output_path = output_dir / safe_filename
        render_args.append((document, output_path))
    
    # Statistics
    stats = {
        "total": len(documents),
        "success": 0,
        "failed": 0,
        "errors": [],
    }
    
    # Render documents
    if parallel and len(documents) > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_render_single_document, args): args[0].id
                for args in render_args
            }
            
            if show_progress:
                futures_iter = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Rendering PDFs",
                    unit="doc"
                )
            else:
                futures_iter = as_completed(futures)
            
            for future in futures_iter:
                success, doc_id, error = future.result()
                if success:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append({"document_id": doc_id, "error": error})
    else:
        # Sequential processing
        if show_progress:
            render_args_iter = tqdm(render_args, desc="Rendering PDFs", unit="doc")
        else:
            render_args_iter = render_args
        
        for args in render_args_iter:
            success, doc_id, error = _render_single_document(args)
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append({"document_id": doc_id, "error": error})
    
    return stats

