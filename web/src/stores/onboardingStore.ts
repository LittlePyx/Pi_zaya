import { create } from 'zustand'
import { appApi, type AppOnboardingStatus } from '../api/app'

interface OnboardingState {
  status: AppOnboardingStatus | null
  loaded: boolean
  pending: boolean
  error: string
  refresh: (force?: boolean) => Promise<void>
}

let onboardingRequestId = 0
let onboardingLastFetchedAt = 0
const ONBOARDING_MIN_REFRESH_MS = 5000

export const useOnboardingStore = create<OnboardingState>((set, get) => ({
  status: null,
  loaded: false,
  pending: false,
  error: '',
  refresh: async (force = false) => {
    if (get().pending) return
    const now = Date.now()
    if (!force && get().loaded && now - onboardingLastFetchedAt < ONBOARDING_MIN_REFRESH_MS) return
    const requestId = ++onboardingRequestId
    set({ pending: true, error: '' })
    try {
      const status = await appApi.onboardingStatus()
      if (requestId !== onboardingRequestId) return
      onboardingLastFetchedAt = Date.now()
      set({ status, loaded: true, pending: false, error: '' })
    } catch (error) {
      if (requestId !== onboardingRequestId) return
      onboardingLastFetchedAt = Date.now()
      set({
        loaded: true,
        pending: false,
        error: error instanceof Error ? error.message : String(error || 'Failed to load onboarding status'),
      })
    }
  },
}))
