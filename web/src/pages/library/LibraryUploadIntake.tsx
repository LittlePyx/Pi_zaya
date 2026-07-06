import { Button, Upload, Typography } from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import './LibraryUploadIntake.css'

const { Dragger } = Upload
const { Text } = Typography

type LibraryUploadIntakeProps = {
  S: Record<string, string>
  uploadLocked: boolean
  uploadDraftCount: number
  showUploadWorkbench: boolean
  lockedMessage: string
  onAddDrafts: (files: File[]) => void
  onToggleWorkbench: () => void
}

export function LibraryUploadIntake({
  S,
  uploadLocked,
  uploadDraftCount,
  showUploadWorkbench,
  lockedMessage,
  onAddDrafts,
  onToggleWorkbench,
}: LibraryUploadIntakeProps) {
  return (
    <section className="kb-lib-workbench-section kb-lib-workbench-section-upload">
      <div className="kb-lib-section-head">
        <div className="kb-lib-section-copy">
          <Text className="kb-lib-section-title">{S.lib_upload_title}</Text>
        </div>
      </div>

      <Dragger
        multiple
        accept=".pdf"
        disabled={uploadLocked}
        showUploadList={false}
        className={`kb-lib-upload-dropzone${uploadLocked ? ' is-locked' : ''}`}
        beforeUpload={(file) => {
          onAddDrafts([file as File])
          return false
        }}
      >
        <div className="kb-lib-upload-dropzone-copy">
          <UploadOutlined className="kb-lib-upload-dropzone-icon" />
          <Text className="kb-lib-upload-dropzone-title">{S.lib_upload_drop_hint}</Text>
          <Text type="secondary" className="kb-lib-upload-dropzone-note">{S.lib_upload_click_hint}</Text>
        </div>
      </Dragger>

      {(uploadDraftCount > 0 || uploadLocked) ? (
        <div className="kb-lib-upload-meta">
          {uploadDraftCount > 0 ? (
            <div className="kb-lib-upload-meta-main">
              <span className="kb-lib-rename-meta">{S.lib_workbench_draft_count.replace('{n}', String(uploadDraftCount))}</span>
              <Button className="kb-lib-action-quiet" onClick={onToggleWorkbench}>
                {showUploadWorkbench ? S.lib_workbench_hide_queue : S.lib_workbench_upload_queue}
              </Button>
            </div>
          ) : null}
          {uploadLocked ? (
            <Text type="secondary" className="kb-lib-upload-inline-note">
              {lockedMessage}
            </Text>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
