// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"fmt"
	"path/filepath"

	"github.com/spf13/cobra"

	"sigs.k8s.io/kustomize/v3/pkg/fs"
	"sigs.k8s.io/kustomize/v3/pkg/transformers/config/defaultconfig"
)

// NewCmdConfig returns an instance of 'config' subcommand.
func NewCmdConfig(fsys fs.FileSystem) *cobra.Command {
	c := &cobra.Command{
		Use:   "config",
		Short: "Config Kustomize transformers",
		Long:  "",
		Example: `
	# Save the default transformer configurations to a local directory
	kustomize config save -d ~/.kustomize/config
`,
		Args: cobra.MinimumNArgs(1),
	}
	c.AddCommand(
		newCmdSave(fsys),
	)
	return c
}

type saveOptions struct {
	saveDirectory string
}

func newCmdSave(fsys fs.FileSystem) *cobra.Command {
	var o saveOptions

	c := &cobra.Command{
		Use:   "save",
		Short: "Save default kustomize transformer configurations to a local directory",
		Long:  "",
		Example: `
	# Save the default transformer configurations to a local directory
	save -d ~/.kustomize/config

`,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate()
			if err != nil {
				return err
			}
			err = o.Complete(fsys)
			if err != nil {
				return err
			}
			return o.RunSave(fsys)
		},
	}
	c.Flags().StringVarP(
		&o.saveDirectory,
		"directory", "d", "",
		"Directory to save the default transformer configurations")

	return c

}

// Validate validates the saveOptions is not empty
func (o *saveOptions) Validate() error {
	if o.saveDirectory == "" {
		return fmt.Errorf("must specify one local directory to save the default transformer configurations")
	}
	return nil
}

// Complete creates the save directory when the directory doesn't exist
func (o *saveOptions) Complete(fsys fs.FileSystem) error {
	if !fsys.Exists(o.saveDirectory) {
		return fsys.MkdirAll(o.saveDirectory)
	}
	if fsys.IsDir(o.saveDirectory) {
		return nil
	}
	return fmt.Errorf("%s is not a directory", o.saveDirectory)
}

// RunSave saves the default transformer configurations local directory
func (o *saveOptions) RunSave(fsys fs.FileSystem) error {
	m := defaultconfig.GetDefaultFieldSpecsAsMap()
	for tname, tcfg := range m {
		filename := filepath.Join(o.saveDirectory, tname+".yaml")
		err := fsys.WriteFile(filename, []byte(tcfg))
		if err != nil {
			return err
		}
	}
	return nil
}
