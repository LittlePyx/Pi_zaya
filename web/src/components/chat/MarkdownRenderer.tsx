import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

function normalize(text: string) {
  return text
    .replace(/\\\(/g, '$').replace(/\\\)/g, '$')
    .replace(/\\\[/g, '$$').replace(/\\\]/g, '$$')
}

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="kb-markdown prose dark:prose-invert max-w-none text-sm">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {normalize(content)}
      </ReactMarkdown>
    </div>
  )
}
