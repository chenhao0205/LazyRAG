package evalset

import (
	"testing"
)

// TestImportValuesEmpty returns true only when all template fields are empty.
func TestImportValuesEmpty(t *testing.T) {
	// All empty
	values := map[string]string{}
	if !importValuesEmpty(values) {
		t.Fatal("empty values should be empty")
	}

	// One field filled
	values2 := map[string]string{"question": "hello"}
	if importValuesEmpty(values2) {
		t.Fatal("non-empty question should not be empty")
	}

	// Whitespace-only
	values3 := map[string]string{"question": "  ", "ground_truth": "\t"}
	if !importValuesEmpty(values3) {
		t.Fatal("whitespace-only should be empty")
	}
}

// TestCleanedCSVHeader strips BOM from the first header value.
func TestCleanedCSVHeader(t *testing.T) {
	header := []string{"\ufeffcase_id", "question", "ground_truth"}
	cleaned := cleanedCSVHeader(header)
	if cleaned[0] != "case_id" {
		t.Fatalf("header[0] = %q, want case_id", cleaned[0])
	}
	if cleaned[1] != "question" {
		t.Fatalf("header[1] = %q", cleaned[1])
	}
}

// TestRowFromValuesWithoutBool maps string values without parsing is_deleted.
func TestRowFromValuesWithoutBool(t *testing.T) {
	values := map[string]string{
		"question":      "what?",
		"ground_truth":  "answer",
		"question_type": "qa",
	}
	row := rowFromValuesWithoutBool(values)
	if row.Question != "what?" {
		t.Fatalf("question = %q", row.Question)
	}
	if row.GroundTruth != "answer" {
		t.Fatalf("ground_truth = %q", row.GroundTruth)
	}
	// is_deleted is not mapped
	if row.IsDeleted {
		t.Fatal("is_deleted should be false")
	}
}

// TestCsvImportValues maps a CSV record to field values via header index.
func TestCsvImportValues(t *testing.T) {
	headerIndex := map[string]int{
		"question":      0,
		"ground_truth":  1,
		"question_type": 2,
	}
	record := []string{"q1", "gt1", "qt1"}
	values := csvImportValues(record, headerIndex)
	if values["question"] != "q1" {
		t.Fatalf("question = %q", values["question"])
	}
	if values["ground_truth"] != "gt1" {
		t.Fatalf("ground_truth = %q", values["ground_truth"])
	}

	// Missing header → empty value
	values2 := csvImportValues(record, map[string]int{})
	if values2["question"] != "" {
		t.Fatalf("missing header should yield empty, got %q", values2["question"])
	}
}

// TestImportTemplateFields covers all expected import fields.
func TestImportTemplateFields(t *testing.T) {
	// Verify required fields exist in template
	hasField := func(field string) bool {
		for _, f := range importTemplateFields {
			if f == field {
				return true
			}
		}
		return false
	}
	for _, required := range importRequiredFields {
		if !hasField(required) {
			t.Fatalf("required field %q not in template fields", required)
		}
	}
}

func TestDatasetImportContractKeepsAllEvoFields(t *testing.T) {
	data := []byte("case_id,question,question_type,difficulty,ground_truth,grading_guidance,key_points,forbidden_claims,reference_context,reference_doc,reference_doc_ids,reference_chunk_ids,generate_reason,is_deleted\n" +
		"case_1,Q,precision,medium,A,Guide,[],[],Context,Doc,doc_1,chunk_1,Imported,false\n")

	result, err := parseCSVImportRows(data)
	if err != nil {
		t.Fatalf("parse csv: %v", err)
	}
	if len(result.rows) != 1 || len(result.invalidRows) != 0 {
		t.Fatalf("unexpected parse result: %#v", result)
	}
	row := result.rows[0]
	if row.Difficulty != "medium" || row.GradingGuidance != "Guide" || row.ForbiddenClaims != "[]" {
		t.Fatalf("evo fields were not preserved: %#v", row)
	}
}

func TestDatasetImportRequiresGuidanceAndSupportedQuestionType(t *testing.T) {
	data := []byte("question,question_type,ground_truth,grading_guidance\nQ,legacy,A,\n")

	result, err := parseCSVImportRows(data)
	if err != nil {
		t.Fatalf("parse csv: %v", err)
	}
	if len(result.invalidRows) != 1 {
		t.Fatalf("expected one invalid row: %#v", result)
	}
	columns := map[string]bool{}
	for _, detail := range result.invalidRows[0].errors {
		columns[detail.Column] = true
	}
	if !columns["grading_guidance"] || !columns["question_type"] {
		t.Fatalf("expected guidance and question type errors: %#v", result.invalidRows[0].errors)
	}
}

func TestDatasetJSONImportAcceptsStructuredAndListFields(t *testing.T) {
	data := []byte(`[{"question":"Q","question_type":"precision","ground_truth":"A","grading_guidance":"G","key_points":[{"statement":"point"}],"forbidden_claims":["wrong"],"reference_context":[{"chunk_id":"chunk-1","text":"context"}],"reference_doc_ids":["doc-1"],"reference_chunk_ids":["chunk-1"]}]`)

	result, err := parseJSONImportRows(data)
	if err != nil {
		t.Fatalf("parse json: %v", err)
	}
	if len(result.rows) != 1 || len(result.invalidRows) != 0 {
		t.Fatalf("unexpected parse result: %#v", result)
	}
	row := result.rows[0]
	if row.KeyPoints != `[{"statement":"point"}]` || row.ForbiddenClaims != `["wrong"]` || row.ReferenceContext != `[{"chunk_id":"chunk-1","text":"context"}]` {
		t.Fatalf("structured fields were not preserved: %#v", row)
	}
	if row.ReferenceDocIDs != "doc-1" || row.ReferenceChunkIDs != "chunk-1" {
		t.Fatalf("list fields were not normalized: %#v", row)
	}
}

// TestCsvValuesByHeader maps a CSV record to values indexed by header names.
func TestCsvValuesByHeader(t *testing.T) {
	header := []string{"question", "ground_truth", "", "question_type"}
	record := []string{"q1", "gt1", "ignored", "qt1"}

	values := csvValuesByHeader(header, record)
	if values["question"] != "q1" {
		t.Fatalf("question = %q", values["question"])
	}
	if values["ground_truth"] != "gt1" {
		t.Fatalf("ground_truth = %q", values["ground_truth"])
	}
	// Empty header name is skipped
	if _, ok := values[""]; ok {
		t.Fatal("empty header should be skipped")
	}
	if values["question_type"] != "qt1" {
		t.Fatalf("question_type = %q", values["question_type"])
	}

	// Record shorter than header → empty string
	values2 := csvValuesByHeader([]string{"a"}, []string{})
	if values2["a"] != "" {
		t.Fatalf("missing value should be empty, got %q", values2["a"])
	}
}
