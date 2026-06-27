/*
Copyright 2026 The KServe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package pod

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

// TestMergeContainerSpecsEnvWithValueFrom reproduces the env-merge corruption:
// a storage initializer that sources credentials via valueFrom (e.g. S3 creds
// from a secret) must not be broken when a ClusterStorageContainer adds env
// vars with literal values. Previously StrategicMergePatch overlaid the env list
// by position, grafting the CSC value onto a credential's valueFrom and emitting
// an env entry with both value and valueFrom, which the API server rejects
// ("valueFrom: may not be specified when value is not empty") so the predictor
// pod never gets created.
func TestMergeContainerSpecsEnvWithValueFrom(t *testing.T) {
	target := &corev1.Container{
		Name:  "storage-initializer",
		Image: "kserve/storage-initializer:v0.18.0",
		Args:  []string{"s3://bucket/model.zip", "/mnt/models"},
		Env: []corev1.EnvVar{
			{
				Name: "AWS_ACCESS_KEY_ID",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{Name: "s3-creds"},
						Key:                  "AWS_ACCESS_KEY_ID",
					},
				},
			},
			{
				Name: "AWS_SECRET_ACCESS_KEY",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{Name: "s3-creds"},
						Key:                  "AWS_SECRET_ACCESS_KEY",
					},
				},
			},
		},
	}
	crd := &corev1.Container{
		Name: "storage-initializer",
		Env: []corev1.EnvVar{
			{Name: "HF_HUB_DISABLE_XET", Value: "1"},
		},
	}

	require.NoError(t, mergeContainerSpecs(target, crd))

	byName := make(map[string]corev1.EnvVar, len(target.Env))
	for _, e := range target.Env {
		byName[e.Name] = e
	}

	// The bug: no entry may carry both value and valueFrom.
	for _, e := range target.Env {
		if e.Value != "" {
			assert.Nilf(t, e.ValueFrom, "env %q must not set both value and valueFrom", e.Name)
		}
	}

	// CSC env added as a clean literal-value entry.
	require.Contains(t, byName, "HF_HUB_DISABLE_XET")
	assert.Equal(t, "1", byName["HF_HUB_DISABLE_XET"].Value)
	assert.Nil(t, byName["HF_HUB_DISABLE_XET"].ValueFrom)

	// Credential env preserved with valueFrom intact and no literal value.
	require.Contains(t, byName, "AWS_ACCESS_KEY_ID")
	assert.Empty(t, byName["AWS_ACCESS_KEY_ID"].Value)
	require.NotNil(t, byName["AWS_ACCESS_KEY_ID"].ValueFrom)
	require.NotNil(t, byName["AWS_ACCESS_KEY_ID"].ValueFrom.SecretKeyRef)
	assert.Equal(t, "AWS_ACCESS_KEY_ID", byName["AWS_ACCESS_KEY_ID"].ValueFrom.SecretKeyRef.Key)

	require.Contains(t, byName, "AWS_SECRET_ACCESS_KEY")
	assert.Empty(t, byName["AWS_SECRET_ACCESS_KEY"].Value)
	require.NotNil(t, byName["AWS_SECRET_ACCESS_KEY"].ValueFrom)

	// Non-env fields from the target survive the strategic merge unchanged.
	assert.Equal(t, "storage-initializer", target.Name)
	assert.Equal(t, "kserve/storage-initializer:v0.18.0", target.Image)
	assert.Equal(t, []string{"s3://bucket/model.zip", "/mnt/models"}, target.Args)
}

// TestMergeContainerSpecsEnvOverrideByName checks that a CSC env var with the
// same name as an existing literal target env var overrides it (no duplicate,
// CSC wins).
func TestMergeContainerSpecsEnvOverrideByName(t *testing.T) {
	target := &corev1.Container{
		Name: "storage-initializer",
		Env: []corev1.EnvVar{
			{Name: "HF_HUB_ENABLE_HF_TRANSFER", Value: "1"},
			{Name: "KEEP_ME", Value: "yes"},
		},
	}
	crd := &corev1.Container{
		Name: "storage-initializer",
		Env: []corev1.EnvVar{
			{Name: "HF_HUB_ENABLE_HF_TRANSFER", Value: "0"},
		},
	}

	require.NoError(t, mergeContainerSpecs(target, crd))

	byName := make(map[string]corev1.EnvVar, len(target.Env))
	count := map[string]int{}
	for _, e := range target.Env {
		byName[e.Name] = e
		count[e.Name]++
	}

	assert.Equal(t, 1, count["HF_HUB_ENABLE_HF_TRANSFER"], "must not duplicate overridden env var")
	assert.Equal(t, "0", byName["HF_HUB_ENABLE_HF_TRANSFER"].Value, "CSC value must win")
	assert.Equal(t, "yes", byName["KEEP_ME"].Value, "unrelated target env must be preserved")
}

// TestMergeEnvVarsByName exercises the helper directly.
func TestMergeEnvVarsByName(t *testing.T) {
	base := []corev1.EnvVar{
		{Name: "A", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Key: "A"}}},
		{Name: "B", Value: "base-b"},
	}
	overrides := []corev1.EnvVar{
		{Name: "B", Value: "override-b"},
		{Name: "C", Value: "1"},
	}

	merged := mergeEnvVarsByName(base, overrides)

	byName := make(map[string]corev1.EnvVar, len(merged))
	for _, e := range merged {
		byName[e.Name] = e
	}

	require.Len(t, merged, 3)
	// A untouched: valueFrom kept, no literal value.
	require.NotNil(t, byName["A"].ValueFrom)
	assert.Empty(t, byName["A"].Value)
	// B overridden by literal value.
	assert.Equal(t, "override-b", byName["B"].Value)
	assert.Nil(t, byName["B"].ValueFrom)
	// C appended.
	assert.Equal(t, "1", byName["C"].Value)

	// Empty overrides return base unchanged.
	assert.Equal(t, base, mergeEnvVarsByName(base, nil))
}
