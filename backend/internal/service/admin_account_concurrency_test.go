//go:build unit

package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeAccountConcurrencyDefaultsInvalidGrokOAuthToOne(t *testing.T) {
	require.Equal(t, 1, normalizeAccountConcurrency(PlatformGrok, AccountTypeOAuth, 0))
	require.Equal(t, 1, normalizeAccountConcurrency(PlatformGrok, AccountTypeOAuth, -5))
}

func TestNormalizeAccountConcurrencyPreservesExplicitValues(t *testing.T) {
	require.Equal(t, 50, normalizeAccountConcurrency(PlatformGrok, AccountTypeOAuth, 50))
	require.Equal(t, 2, normalizeAccountConcurrency(PlatformOpenAI, AccountTypeOAuth, 2))
	require.Equal(t, 2, normalizeAccountConcurrency(PlatformGrok, AccountTypeAPIKey, 2))
}

func TestGrokOAuthSchedulingConcurrencyUsesConfiguredValue(t *testing.T) {
	account := &Account{Platform: PlatformGrok, Type: AccountTypeOAuth, Concurrency: 24}
	require.Equal(t, 24, account.SchedulingSlotConcurrency())
	require.Equal(t, 24, account.SchedulingLoadConcurrency())
}

func TestGrokOAuthSchedulingConcurrencyDefaultsInvalidValuesToOne(t *testing.T) {
	for _, concurrency := range []int{0, -5} {
		account := &Account{Platform: PlatformGrok, Type: AccountTypeOAuth, Concurrency: concurrency}
		require.Equal(t, 1, account.SchedulingSlotConcurrency())
		require.Equal(t, 1, account.SchedulingLoadConcurrency())
	}
}
