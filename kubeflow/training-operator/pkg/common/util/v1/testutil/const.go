// Copyright 2018 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutil

import (
	"time"
)

const (
	TestImageName = "test-image-for-kubeflow-training-operator:latest"
	TestTFJobName = "test-tfjob"
	LabelWorker   = "worker"
	LabelPS       = "ps"
	LabelChief    = "chief"
	TFJobKind     = "TFJob"

	SleepInterval = 500 * time.Millisecond
	ThreadCount   = 1
)

var (
	AlwaysReady = func() bool { return true }
)
