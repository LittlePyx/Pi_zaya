"""
Markdown Quality Analyzer

Analyzes generated markdown to identify potential issues and improvements.
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityIssue:
    """Represents a quality issue found in the markdown."""
    category: str  # 'heading', 'formula', 'table', 'reference', 'noise', etc.
    severity: str  # 'error', 'warning', 'info'
    message: str
    line_number: int
    suggestion: str = ""


class MarkdownAnalyzer:
    """Analyzes markdown quality and suggests improvements."""
    
    def __init__(self):
        self.issues: List[QualityIssue] = []
        self.heading_levels: List[int] = []
        self.reference_section_start: int = -1
        
    def analyze(self, md_content: str, md_path: Path | None = None) -> List[QualityIssue]:
        """Analyze markdown content and return list of issues."""
        self.issues = []
        self.heading_levels = []
        self.reference_section_start = -1
        
        lines = md_content.splitlines()
        
        # Analyze headings
        self._analyze_headings(lines)
        
        # Analyze formulas
        self._analyze_formulas(lines)
        
        # Analyze tables
        self._analyze_tables(lines)
        
        # Analyze references
        self._analyze_references(lines)
        
        # Analyze noise
        self._analyze_noise(lines)
        
        # Analyze structure
        self._analyze_structure(lines)
        
        # Analyze captions
        self._analyze_captions(lines)
        
        return self.issues
    
    def _analyze_headings(self, lines: List[str]) -> None:
        """Analyze heading hierarchy and consistency."""
        current_levels = [0]  # Track heading levels
        
        for i, line in enumerate(lines, 1):
            # Check for markdown headings
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                
                # Check if heading looks like a formula (common mistake)
                if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text) or \
                   re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text) or \
                   re.search(r'^\s*[a-z]\s*\d+', text) or \
                   (len(text) <= 10 and re.search(r'\d+.*[a-z]|[a-z].*\d+', text) and not re.search(r'[A-Z]{2,}', text)):
                    self.issues.append(QualityIssue(
                        category='heading',
                        severity='error',
                        message=f'Heading looks like a formula: "{text[:50]}"',
                        line_number=i,
                        suggestion='This should be classified as math, not heading. Check formula detection.'
                    ))
                    continue
                
                # Check for skipped levels (e.g., H1 -> H3)
                if level > current_levels[-1] + 1:
                    self.issues.append(QualityIssue(
                        category='heading',
                        severity='warning',
                        message=f'Heading level skipped: H{current_levels[-1]} -> H{level}',
                        line_number=i,
                        suggestion=f'Consider using H{current_levels[-1] + 1} instead'
                    ))
                
                # Update current levels
                while len(current_levels) > 0 and current_levels[-1] >= level:
                    current_levels.pop()
                current_levels.append(level)
                self.heading_levels.append(level)
                
                # Check for very long headings
                if len(text) > 100:
                    self.issues.append(QualityIssue(
                        category='heading',
                        severity='info',
                        message=f'Very long heading: {len(text)} characters',
                        line_number=i,
                        suggestion='Consider shortening or splitting'
                    ))
                
                # Check for headings that look like body text
                if len(text.split()) > 15:
                    self.issues.append(QualityIssue(
                        category='heading',
                        severity='warning',
                        message=f'Heading looks like body text: {len(text.split())} words',
                        line_number=i,
                        suggestion='Verify this is actually a heading'
                    ))
                
                # Check for headings that look like captions
                if re.match(r'^\s*(?:Fig\.|Figure|Table|Algorithm)\s*(?:\d+|[IVXLC]+)', text, re.IGNORECASE):
                    self.issues.append(QualityIssue(
                        category='heading',
                        severity='error',
                        message=f'Heading looks like a caption: "{text[:50]}"',
                        line_number=i,
                        suggestion='This should be classified as caption, not heading.'
                    ))
    
    def _analyze_formulas(self, lines: List[str]) -> None:
        """Analyze formula formatting."""
        in_formula = False
        formula_start = 0
        formula_count = 0
        potential_formulas_as_headings = []
        
        for i, line in enumerate(lines, 1):
            # Check for block formulas ($$)
            if re.match(r'^\s*\$\$\s*$', line):
                if in_formula:
                    # End of formula
                    formula_lines = i - formula_start
                    if formula_lines < 2:
                        self.issues.append(QualityIssue(
                            category='formula',
                            severity='warning',
                            message='Very short formula block',
                            line_number=formula_start,
                            suggestion='Verify formula is complete'
                        ))
                    in_formula = False
                    formula_count += 1
                else:
                    in_formula = True
                    formula_start = i
            
            # Check for inline formulas
            if re.search(r'\$[^$]+\$', line) and not in_formula:
                # Check for unclosed formulas
                dollar_count = line.count('$')
                if dollar_count % 2 != 0:
                    self.issues.append(QualityIssue(
                        category='formula',
                        severity='error',
                        message='Unclosed inline formula',
                        line_number=i,
                        suggestion='Check for missing $ delimiter'
                    ))
            
            # Check for formulas that might have been misclassified as headings
            if re.match(r'^#+\s+', line):
                text = re.sub(r'^#+\s+', '', line).strip()
                # Check if it looks like a formula
                if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text) or \
                   re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text) or \
                   (len(text) <= 15 and re.search(r'\d+.*[a-z]|[a-z].*\d+', text) and '=' in text):
                    potential_formulas_as_headings.append((i, text[:50]))
        
        if potential_formulas_as_headings:
            self.issues.append(QualityIssue(
                category='formula',
                severity='error',
                message=f'Found {len(potential_formulas_as_headings)} formulas misclassified as headings',
                line_number=potential_formulas_as_headings[0][0],
                suggestion='These should be in $$ blocks, not headings. Check formula detection logic.'
            ))
        
        # Check formula count
        if formula_count == 0 and len(lines) > 50:
            # Large document with no formulas might indicate detection issues
            self.issues.append(QualityIssue(
                category='formula',
                severity='info',
                message='No formulas detected in document',
                line_number=1,
                suggestion='If document contains math, check formula detection'
            ))
    
    def _analyze_tables(self, lines: List[str]) -> None:
        """Analyze table formatting."""
        in_table = False
        table_start = 0
        table_rows = []
        
        for i, line in enumerate(lines, 1):
            if '|' in line and not line.strip().startswith('```'):
                if not in_table:
                    in_table = True
                    table_start = i
                    table_rows = []
                table_rows.append(line)
            else:
                if in_table:
                    # End of table
                    if len(table_rows) < 2:
                        self.issues.append(QualityIssue(
                            category='table',
                            severity='warning',
                            message='Table with less than 2 rows',
                            line_number=table_start,
                            suggestion='Verify table is complete'
                        ))
                    else:
                        # Check column consistency
                        col_counts = [len(row.split('|')) for row in table_rows]
                        if len(set(col_counts)) > 1:
                            self.issues.append(QualityIssue(
                                category='table',
                                severity='error',
                                message=f'Inconsistent column counts: {col_counts}',
                                line_number=table_start,
                                suggestion='Fix table alignment'
                            ))
                    in_table = False
    
    def _analyze_references(self, lines: List[str]) -> None:
        """Analyze references section."""
        in_references = False
        ref_start = 0
        ref_count = 0
        
        for i, line in enumerate(lines, 1):
            # Check for REFERENCES heading
            if re.match(r'^#+\s*REFERENCES?\s*$', line, re.IGNORECASE):
                in_references = True
                ref_start = i
                self.reference_section_start = i
                continue
            
            if in_references:
                # Count reference-like lines
                # References typically have: author, year, title, journal, etc.
                if re.search(r'\b(?:19|20)\d{2}\b', line):  # Year
                    if re.search(r'\b(?:doi|http|arxiv|vol\.|pp\.|pages?)\b', line, re.IGNORECASE):
                        ref_count += 1
                    elif len(line.strip()) > 30 and ',' in line:
                        ref_count += 1
        
        if in_references and ref_count < 5:
            self.issues.append(QualityIssue(
                category='reference',
                severity='warning',
                message=f'References section has only {ref_count} potential references',
                line_number=ref_start,
                suggestion='Verify references were extracted correctly'
            ))
    
    def _analyze_noise(self, lines: List[str]) -> None:
        """Analyze for noise patterns."""
        noise_patterns = [
            (r'^ACM Trans\.', 'ACM journal header'),
            (r'^Publication date:', 'Publication metadata'),
            (r'^PDF Download', 'PDF download link'),
            (r'^Total Citations:', 'Citation count'),
            (r'^RESEARCH-ARTICLE', 'Article type marker'),
            (r'^Latest updates:', 'Update notice'),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, desc in noise_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    self.issues.append(QualityIssue(
                        category='noise',
                        severity='warning',
                        message=f'Potential noise detected: {desc}',
                        line_number=i,
                        suggestion='Consider removing this line'
                    ))
    
    def _analyze_captions(self, lines: List[str]) -> None:
        """Analyze figure and table captions."""
        captions_found = []
        images_found = []
        
        for i, line in enumerate(lines, 1):
            # Check for images
            if re.search(r'!\[.*?\]\(.*?\)', line):
                images_found.append(i)
            
            # Check for captions (italicized text starting with Fig./Table)
            if re.match(r'^\s*\*.*?(?:Fig\.|Figure|Table|Algorithm)\s*(?:\d+|[IVXLC]+)', line, re.IGNORECASE):
                captions_found.append(i)
            # Also check non-italicized captions
            elif re.match(r'^\s*(?:Fig\.|Figure|Table|Algorithm)\s*(?:\d+|[IVXLC]+)', line, re.IGNORECASE):
                # Check if it's not a heading
                if not re.match(r'^#+\s+', line):
                    captions_found.append(i)
                    self.issues.append(QualityIssue(
                        category='caption',
                        severity='info',
                        message=f'Caption found but not italicized: "{line[:60]}"',
                        line_number=i,
                        suggestion='Captions should be italicized (*text*)'
                    ))
        
        # Check if images have nearby captions
        for img_line in images_found:
            # Look for caption within 3 lines after image
            has_caption = False
            for j in range(img_line, min(img_line + 4, len(lines) + 1)):
                if j in captions_found:
                    has_caption = True
                    break
            if not has_caption:
                self.issues.append(QualityIssue(
                    category='caption',
                    severity='warning',
                    message=f'Image at line {img_line} has no nearby caption',
                    line_number=img_line,
                    suggestion='Consider adding a caption for the figure'
                ))
        
        if len(images_found) > 0 and len(captions_found) == 0:
            self.issues.append(QualityIssue(
                category='caption',
                severity='warning',
                message=f'Found {len(images_found)} images but no captions',
                line_number=images_found[0] if images_found else 1,
                suggestion='Check if captions were properly extracted'
            ))
    
    def _analyze_structure(self, lines: List[str]) -> None:
        """Analyze overall document structure."""
        # Check for very long paragraphs
        current_para = []
        para_start = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                if len(current_para) > 20:
                    self.issues.append(QualityIssue(
                        category='structure',
                        severity='info',
                        message=f'Very long paragraph: {len(current_para)} lines',
                        line_number=para_start,
                        suggestion='Consider breaking into smaller paragraphs'
                    ))
                current_para = []
            elif not stripped.startswith('#') and not stripped.startswith('|') and not stripped.startswith('```'):
                if not current_para:
                    para_start = i
                current_para.append(line)
        
        # Check heading distribution
        if self.heading_levels:
            h1_count = sum(1 for h in self.heading_levels if h == 1)
            if h1_count == 0:
                self.issues.append(QualityIssue(
                    category='structure',
                    severity='info',
                    message='No H1 headings found',
                    line_number=1,
                    suggestion='Consider adding a main title'
                ))
            elif h1_count > 5:
                self.issues.append(QualityIssue(
                    category='structure',
                    severity='warning',
                    message=f'Many H1 headings: {h1_count}',
                    line_number=1,
                    suggestion='Consider using H2/H3 for subsections'
                ))
    
    def generate_report(self) -> str:
        """Generate a human-readable quality report."""
        if not self.issues:
            return "✓ No quality issues detected.\n"
        
        report = ["# Markdown Quality Analysis Report\n"]
        
        by_category = {}
        for issue in self.issues:
            by_category.setdefault(issue.category, []).append(issue)
        
        for category, issues in sorted(by_category.items()):
            report.append(f"\n## {category.upper()} Issues ({len(issues)})\n")
            for issue in issues:
                severity_icon = {
                    'error': '❌',
                    'warning': '⚠️',
                    'info': 'ℹ️'
                }.get(issue.severity, '•')
                report.append(
                    f"{severity_icon} **Line {issue.line_number}** ({issue.severity}): "
                    f"{issue.message}"
                )
                if issue.suggestion:
                    report.append(f"   💡 Suggestion: {issue.suggestion}")
                report.append("")
        
        return "\n".join(report)
