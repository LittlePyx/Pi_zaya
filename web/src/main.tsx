import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { installUserIssueReporter } from './userIssueReporter'
import './styles/index.css'
import './styles/chat.css'
import './styles/library.css'
import './styles/auth.css'

installUserIssueReporter()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
