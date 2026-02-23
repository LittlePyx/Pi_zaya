import { Typography } from 'antd'
import { useChatStore } from '../stores/chatStore'
import { useSettingsStore } from '../stores/settingsStore'
import { MessageList } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import { RefsPanel } from '../components/refs/RefsPanel'
import { S } from '../i18n/zh'

const { Text } = Typography

export default function ChatPage() {
  const messages = useChatStore(s => s.messages)
  const generation = useChatStore(s => s.generation)
  const refs = useChatStore(s => s.refs)
  const activeConvId = useChatStore(s => s.activeConvId)
  const sendMessage = useChatStore(s => s.sendMessage)
  const cancelGen = useChatStore(s => s.cancelGeneration)
  const settings = useSettingsStore()

  const onSend = (text: string) => {
    sendMessage(text, {
      topK: settings.topK,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      deepRead: settings.deepRead,
    })
  }

  // Find user message IDs for refs display
  const userMsgIds = messages.filter(m => m.role === 'user').map(m => m.id)

  return (
    <div className="flex flex-col h-full">
      {!activeConvId && messages.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <Text type="secondary">{S.no_msgs}</Text>
        </div>
      ) : (
        <>
          <MessageList
            messages={messages}
            generationPartial={generation?.partial}
            generationStage={generation?.stage}
          />
          {/* Refs panels for each user message */}
          {userMsgIds.map(id => (
            <RefsPanel key={id} refs={refs} msgId={id} />
          ))}
        </>
      )}
      <ChatInput onSend={onSend} onStop={cancelGen} generating={!!generation} />
    </div>
  )
}
