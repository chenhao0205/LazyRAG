package chat

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"lazymind/core/doc"
	"lazymind/core/log"
)

// Keep a Core-side limit even when a channel gateway has already limited media.
// It protects direct Core callers and bounds base64 decoding allocations.
const maxInputBase64AttachmentBytes = 25 * 1024 * 1024

var inputBase64ImageExtensions = map[string]string{
	"image/jpeg": ".jpg",
	"image/png":  ".png",
	"image/gif":  ".gif",
	"image/webp": ".webp",
}

var inputBase64FileExtensions = map[string]string{
	"text/plain":      ".txt",
	"text/markdown":   ".md",
	"application/pdf": ".pdf",
}

// materializeInputBase64Attachments converts accepted attachment data URLs into
// the same upload-root file paths used by regular chat uploads. It mutates input
// entries to use uri, so both the current turn and persisted history follow the
// existing files_per_turn path.
func materializeInputBase64Attachments(raw map[string]any) {
	input, ok := raw["input"].([]any)
	if !ok {
		return
	}
	for _, item := range input {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		typ, _ := entry["input_type"].(string)
		typ = strings.ToLower(strings.TrimSpace(typ))
		if typ != "image" && typ != "file" {
			continue
		}
		if uri, _ := entry["uri"].(string); strings.TrimSpace(uri) != "" {
			continue
		}
		dataURL, _ := entry["input_base64"].(string)
		if strings.TrimSpace(dataURL) == "" {
			continue
		}

		path, err := materializeAttachmentDataURL(typ, dataURL)
		if err != nil {
			// Do not retain untrusted base64 in history or allow it to reach an
			// alternative consumer after this bridge has rejected it.
			delete(entry, "input_base64")
			log.Logger.Warn().Err(err).Str("input_type", typ).Msg("rejecting chat input_base64 attachment")
			continue
		}
		entry["uri"] = path
		delete(entry, "input_base64")
	}
}

func materializeAttachmentDataURL(inputType, dataURL string) (string, error) {
	mimeType, encoded, err := parseAttachmentDataURL(dataURL)
	if err != nil {
		return "", err
	}
	extension := ".bin"
	if inputType == "image" {
		var ok bool
		extension, ok = inputBase64ImageExtensions[mimeType]
		if !ok {
			return "", fmt.Errorf("unsupported image data URL MIME type %q", mimeType)
		}
	} else if inputType == "file" {
		if knownExtension, ok := inputBase64FileExtensions[mimeType]; ok {
			extension = knownExtension
		}
	} else {
		return "", fmt.Errorf("unsupported attachment input type %q", inputType)
	}
	if len(encoded) > base64.StdEncoding.EncodedLen(maxInputBase64AttachmentBytes+1) {
		return "", fmt.Errorf("attachment data URL exceeds %d byte limit", maxInputBase64AttachmentBytes)
	}
	decoded, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil {
		return "", fmt.Errorf("decode attachment data URL: %w", err)
	}
	if len(decoded) == 0 || len(decoded) > maxInputBase64AttachmentBytes {
		return "", fmt.Errorf("attachment data URL decoded size is outside allowed range")
	}
	if inputType == "image" && !matchesImageMagic(mimeType, decoded) {
		return "", fmt.Errorf("image data URL bytes do not match MIME type %q", mimeType)
	}

	directory := filepath.Join(doc.UploadRoot(), "tmp", "chat-input-base64")
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", fmt.Errorf("create chat attachment directory: %w", err)
	}
	var token [16]byte
	if _, err := rand.Read(token[:]); err != nil {
		return "", fmt.Errorf("generate chat attachment name: %w", err)
	}
	path := filepath.Join(directory, hex.EncodeToString(token[:])+extension)
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", fmt.Errorf("create chat attachment: %w", err)
	}
	if n, err := f.Write(decoded); err != nil || n != len(decoded) {
		_ = f.Close()
		_ = os.Remove(path)
		if err == nil {
			err = fmt.Errorf("short write: wrote %d of %d bytes", n, len(decoded))
		}
		return "", fmt.Errorf("write chat attachment: %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return "", fmt.Errorf("close chat attachment: %w", err)
	}
	return path, nil
}

func parseAttachmentDataURL(dataURL string) (mimeType, encoded string, err error) {
	if !strings.HasPrefix(dataURL, "data:") {
		return "", "", fmt.Errorf("attachment input_base64 is not a data URL")
	}
	comma := strings.IndexByte(dataURL, ',')
	if comma < 0 {
		return "", "", fmt.Errorf("attachment data URL is missing payload")
	}
	meta := strings.TrimPrefix(dataURL[:comma], "data:")
	parts := strings.Split(meta, ";")
	if len(parts) != 2 || !strings.EqualFold(strings.TrimSpace(parts[1]), "base64") {
		return "", "", fmt.Errorf("attachment data URL must use base64 encoding")
	}
	parsed, _, parseErr := mime.ParseMediaType(strings.TrimSpace(parts[0]))
	if parseErr != nil || parsed == "" || !strings.Contains(parsed, "/") {
		return "", "", fmt.Errorf("attachment data URL has invalid MIME type")
	}
	mimeType = strings.ToLower(parsed)
	encoded = dataURL[comma+1:]
	if encoded == "" {
		return "", "", fmt.Errorf("attachment data URL payload is empty")
	}
	return mimeType, encoded, nil
}

func matchesImageMagic(mimeType string, data []byte) bool {
	switch mimeType {
	case "image/jpeg":
		return len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff
	case "image/png":
		return len(data) >= 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n"
	case "image/gif":
		return len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a")
	case "image/webp":
		return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP"
	default:
		return false
	}
}
