import { create } from 'zustand'
import { appApi, type AppOnboardingStatus } from '../api/app'

interface OnboardingState {
  status: AppOnboardingStatus | null
  loaded: boolean
  pending: boolean
  error: string
  refresh: () => Promise<void>
}

let onboardingRequestId = 0

export const useOnboardingStore = create<OnboardingState>((set, get) => ({
  status: null,
  loaded: false,
  pending: false,
  error: '',
  refresh: async () => {
    if (get().pending) return
    const requestId = ++onboardingRequestId
    set({ pending: true, error: '' })
    try {
      const status = await appApi.onboardingStatus()
      if (requestId !== onboardingRequestId) return
      set({ status, loaded: true, pending: false, error: '' })
    } catch (error) {
      if (requestId !== onboardingRequestId) return
      set({
        loaded: true,
        pending: false,
        error: error instanceof Error ? error.message : String(error || 'Failed to load onboarding status'),
      })
    }
  },
}))
