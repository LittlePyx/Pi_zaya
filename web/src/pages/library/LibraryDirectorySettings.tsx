import { Button, Input, Typography } from 'antd'
import { FolderOpenOutlined, SaveOutlined } from '@ant-design/icons'
import './LibraryDirectorySettings.css'

const { Text } = Typography

type LibraryDirectoryKind = 'pdf' | 'md'
type LibraryDirectoryOpenTarget = 'pdf_dir' | 'md_dir'

type LibraryDirectorySettingsProps = {
  S: Record<string, string>
  directoriesConfigured: boolean
  showDirEditor: boolean
  pdfDirDraft: string
  mdDirDraft: string
  pickingDir: LibraryDirectoryKind | null
  savingDirs: boolean
  dirDirty: boolean
  onToggleEditor: () => void
  onPdfDirChange: (value: string) => void
  onMdDirChange: (value: string) => void
  onPickDir: (target: LibraryDirectoryKind) => void | Promise<unknown>
  onOpenFolder: (target: LibraryDirectoryOpenTarget) => void | Promise<unknown>
  onSaveDirs: () => void | Promise<unknown>
}

export function LibraryDirectorySettings({
  S,
  directoriesConfigured,
  showDirEditor,
  pdfDirDraft,
  mdDirDraft,
  pickingDir,
  savingDirs,
  dirDirty,
  onToggleEditor,
  onPdfDirChange,
  onMdDirChange,
  onPickDir,
  onOpenFolder,
  onSaveDirs,
}: LibraryDirectorySettingsProps) {
  return (
    <section className="kb-lib-workbench-section">
      <div className="kb-lib-section-head">
        <div className="kb-lib-section-copy">
          <Text className="kb-lib-section-title">{S.lib_section_dir}</Text>
        </div>
        {directoriesConfigured ? (
          <Button className="kb-lib-action-quiet" onClick={onToggleEditor}>
            {showDirEditor ? S.lib_dir_collapse : S.lib_dir_edit}
          </Button>
        ) : null}
      </div>

      <div className="kb-lib-dir-summary">
        <div className={`kb-lib-dir-summary-row${showDirEditor ? ' is-editing' : ''}`}>
          <Text className="kb-lib-dir-summary-label">PDF</Text>
          {showDirEditor ? (
            <Input
              value={pdfDirDraft}
              placeholder={S.lib_dir_select_pdf}
              onChange={(event) => onPdfDirChange(event.target.value)}
            />
          ) : (
            <Text className="kb-lib-dir-summary-path" ellipsis={{ tooltip: pdfDirDraft || S.lib_dir_no_pdf }}>
              {pdfDirDraft || S.lib_dir_no_pdf}
            </Text>
          )}
          {showDirEditor ? (
            <Button
              className="kb-lib-action-quiet"
              loading={pickingDir === 'pdf'}
              onClick={() => { void onPickDir('pdf') }}
            >
              {S.lib_dir_pick}
            </Button>
          ) : null}
          <Button
            className="kb-lib-action-quiet"
            icon={<FolderOpenOutlined />}
            disabled={!pdfDirDraft.trim()}
            onClick={() => { void onOpenFolder('pdf_dir') }}
          >
            {S.lib_dir_open}
          </Button>
        </div>
        <div className={`kb-lib-dir-summary-row${showDirEditor ? ' is-editing' : ''}`}>
          <Text className="kb-lib-dir-summary-label">MD</Text>
          {showDirEditor ? (
            <Input
              value={mdDirDraft}
              placeholder={S.lib_dir_select_md}
              onChange={(event) => onMdDirChange(event.target.value)}
            />
          ) : (
            <Text className="kb-lib-dir-summary-path" ellipsis={{ tooltip: mdDirDraft || S.lib_dir_no_md }}>
              {mdDirDraft || S.lib_dir_no_md}
            </Text>
          )}
          {showDirEditor ? (
            <Button
              className="kb-lib-action-quiet"
              loading={pickingDir === 'md'}
              onClick={() => { void onPickDir('md') }}
            >
              {S.lib_dir_pick}
            </Button>
          ) : null}
          <Button
            className="kb-lib-action-quiet"
            icon={<FolderOpenOutlined />}
            disabled={!mdDirDraft.trim()}
            onClick={() => { void onOpenFolder('md_dir') }}
          >
            {S.lib_dir_open}
          </Button>
        </div>
      </div>

      {showDirEditor ? (
        <div className="kb-lib-section-actions">
          <Button
            className="kb-lib-action-tonal"
            type="primary"
            icon={<SaveOutlined />}
            loading={savingDirs}
            disabled={!dirDirty}
            onClick={() => { void onSaveDirs() }}
          >
            {S.lib_dir_save}
          </Button>
        </div>
      ) : null}
    </section>
  )
}
