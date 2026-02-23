import { useEffect, useState } from 'react'
import { Upload, Button, List, message, Progress, Select, Typography } from 'antd'
import { UploadOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons'
import { useLibraryStore } from '../stores/libraryStore'
import { S } from '../i18n/zh'

const { Text } = Typography
const { Dragger } = Upload

export default function LibraryPage() {
  const store = useLibraryStore()
  const [mode, setMode] = useState('balanced')

  useEffect(() => { store.loadPdfs() }, [])

  const handleUpload = async (file: File) => {
    const res = await store.upload(file)
    if (res.duplicate) {
      message.warning(`${S.dup_found}: ${res.existing}`)
    } else {
      message.success(`已保存: ${res.name}`)
      store.loadPdfs()
    }
    return false
  }

  const handleReindex = async () => {
    const hide = message.loading('正在更新知识库...', 0)
    const res = await store.reindex()
    hide()
    message[res.ok ? 'success' : 'error'](res.ok ? S.run_ok : S.run_fail)
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <Text className="text-xl font-bold">{S.page_library}</Text>

      <Dragger
        multiple
        accept=".pdf"
        showUploadList={false}
        beforeUpload={handleUpload as never}
      >
        <p className="text-lg"><UploadOutlined /></p>
        <p>{S.upload_pdf}（{S.name_rule}）</p>
      </Dragger>

      <div className="flex gap-2 items-center">
        <Select value={mode} onChange={setMode} className="w-40" options={[
          { value: 'balanced', label: '均衡' },
          { value: 'ultra_fast', label: '极速' },
          { value: 'no_llm', label: '无 LLM' },
        ]} />
        <Button onClick={() => {
          store.pdfs.forEach(p => store.convert(p.name, mode))
        }}>{S.convert_now}</Button>
        {store.converting && (
          <Button icon={<StopOutlined />} danger onClick={() => store.cancelConvert()}>停止</Button>
        )}
      </div>

      {store.converting && <Progress percent={50} status="active" />}

      <Button icon={<ReloadOutlined />} type="primary" onClick={handleReindex}>
        {S.reindex_now}
      </Button>

      <List
        size="small"
        header={<Text strong>PDF 文件 ({store.pdfs.length})</Text>}
        dataSource={store.pdfs}
        renderItem={p => (
          <List.Item>
            <Text className="text-sm">{p.name}</Text>
          </List.Item>
        )}
      />
    </div>
  )
}
