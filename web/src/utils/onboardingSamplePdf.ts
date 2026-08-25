type SamplePdfLine = {
  text: string
  y: number
  size?: number
  bold?: boolean
}

const SAMPLE_FILE_NAME = 'Pi_zaya-Getting-Started-Sample.pdf'
const SAMPLE_LAST_MODIFIED = Date.UTC(2026, 0, 1)

function escapePdfText(value: string) {
  return value
    .replace(/\\/g, '\\\\')
    .replace(/\(/g, '\\(')
    .replace(/\)/g, '\\)')
}

function pageStream(lines: SamplePdfLine[]) {
  return lines
    .map((line) => {
      const font = line.bold ? 'F2' : 'F1'
      const size = line.size || 11
      return `BT /${font} ${size} Tf 72 ${line.y} Td (${escapePdfText(line.text)}) Tj ET`
    })
    .join('\n')
}

function streamObject(stream: string) {
  return `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`
}

function buildPdf(objects: string[]) {
  let pdf = '%PDF-1.4\n'
  const offsets = [0]
  objects.forEach((object, index) => {
    offsets[index + 1] = pdf.length
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`
  })
  const xrefOffset = pdf.length
  pdf += `xref\n0 ${objects.length + 1}\n`
  pdf += '0000000000 65535 f \n'
  for (let index = 1; index <= objects.length; index += 1) {
    pdf += `${String(offsets[index]).padStart(10, '0')} 00000 n \n`
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\n`
  pdf += `startxref\n${xrefOffset}\n%%EOF\n`
  return pdf
}

export function createOnboardingSamplePdf() {
  const pageOne = pageStream([
    { text: 'Pi_zaya Getting Started Sample', y: 744, size: 18, bold: true },
    { text: 'A self-authored demonstration paper for source-grounded reading', y: 720, size: 10 },
    { text: 'Pi_zaya Product Team | 2026', y: 702, size: 10 },
    { text: 'Abstract', y: 666, size: 14, bold: true },
    { text: 'Research assistants can shorten literature review time, but their answers remain useful', y: 644 },
    { text: 'only when important claims can be checked against the source. This demonstration paper', y: 628 },
    { text: 'describes a small workflow for conversion, retrieval, answering, and source verification.', y: 612 },
    { text: '1. Research question', y: 574, size: 14, bold: true },
    { text: 'How can an AI research assistant keep generated answers traceable to source material?', y: 552 },
    { text: 'The practical goal is to reduce search time while preserving a clear path back to the', y: 536 },
    { text: 'passage, section, table, or reference that supports each important statement.', y: 520 },
    { text: '2. Workflow', y: 482, size: 14, bold: true },
    { text: 'Step 1 - Convert the PDF into structured Markdown while retaining page boundaries.', y: 460 },
    { text: 'Step 2 - Retrieve passages that match the question and its research intent.', y: 444 },
    { text: 'Step 3 - Generate an answer from the retrieved evidence and attach source links.', y: 428 },
    { text: 'Step 4 - Let the reader open the cited passage and inspect the surrounding context.', y: 412 },
    { text: 'The answer should state uncertainty when the available passages do not support a claim [1].', y: 388 },
    { text: '3. Findings', y: 350, size: 14, bold: true },
    { text: 'Source links make verification faster because the reader does not need to search the whole', y: 328 },
    { text: 'paper again. Structured conversion also makes headings, equations, and references easier', y: 312 },
    { text: 'to retrieve. Clear uncertainty messages reduce the risk of treating an inference as evidence.', y: 296 },
  ])
  const pageTwo = pageStream([
    { text: '4. Recommended answer structure', y: 744, size: 14, bold: true },
    { text: 'A useful first answer contains four parts: the research question, the method, the main', y: 720 },
    { text: 'findings, and the limitations. Important statements should carry citations that open the', y: 704 },
    { text: 'supporting source passage. Related papers may be suggested separately from direct evidence.', y: 688 },
    { text: '5. Limitations', y: 650, size: 14, bold: true },
    { text: 'Traceability does not guarantee that an interpretation is correct. Conversion errors, weak', y: 628 },
    { text: 'retrieval, or ambiguous writing can still affect an answer. Readers should inspect the cited', y: 612 },
    { text: 'context before relying on a claim in high-stakes research decisions.', y: 596 },
    { text: '6. Conclusion', y: 558, size: 14, bold: true },
    { text: 'A source-grounded workflow turns an AI answer into a reviewable research aid. The shortest', y: 536 },
    { text: 'successful path is simple: prepare one paper, ask one focused question, and open one citation.', y: 520 },
    { text: 'References', y: 470, size: 14, bold: true },
    { text: '[1] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for', y: 448 },
    { text: '    Knowledge-Intensive NLP Tasks. NeurIPS, 2020.', y: 432 },
    { text: '[2] Gao Y, Xiong Y, Gao X, et al. Retrieval-Augmented Generation for Large', y: 408 },
    { text: '    Language Models: A Survey. arXiv:2312.10997, 2023.', y: 392 },
    { text: 'Document note', y: 342, size: 12, bold: true },
    { text: 'This sample was written for Pi_zaya onboarding. It may be copied, converted, and deleted.', y: 322 },
  ])
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 7 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 8 0 R >>',
    streamObject(pageOne),
    streamObject(pageTwo),
  ]
  return new File([buildPdf(objects)], SAMPLE_FILE_NAME, {
    type: 'application/pdf',
    lastModified: SAMPLE_LAST_MODIFIED,
  })
}
