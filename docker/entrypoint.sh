#!/bin/sh
set -eu

die() {
    printf 'audio-jlens container: %s\n' "$*" >&2
    exit 64
}

port="${PORT:-7860}"
case "$port" in
    ''|*[!0-9]*) die "PORT must be an integer between 1 and 65535" ;;
esac
if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    die "PORT must be an integer between 1 and 65535"
fi

lens_path="${JLENS_LENS_PATH:-}"
lens_repo_id="${JLENS_LENS_REPO_ID:-}"
lens_filename="${JLENS_LENS_FILENAME:-}"
phone_signatures_path="${JLENS_PHONE_SIGNATURES_PATH:-}"
phone_signatures_filename="${JLENS_PHONE_SIGNATURES_FILENAME:-}"

if [ -n "$lens_repo_id" ] || [ -n "$lens_filename" ] || \
    [ -n "$phone_signatures_filename" ]; then
    if [ -z "$lens_repo_id" ] || [ -z "$lens_filename" ]; then
        die "JLENS_LENS_REPO_ID and JLENS_LENS_FILENAME must be set together"
    fi
    if [ -n "$lens_path" ]; then
        die "set either JLENS_LENS_PATH or the Hub lens variables, not both"
    fi

    lens_repo_type="${JLENS_LENS_REPO_TYPE:-model}"
    case "$lens_repo_type" in
        model|dataset|space) ;;
        *) die "JLENS_LENS_REPO_TYPE must be model, dataset, or space" ;;
    esac

    printf 'Downloading Jacobian-lens artifact from %s/%s...\n' \
        "$lens_repo_id" "$lens_filename"
    if [ -n "${JLENS_LENS_REVISION:-}" ]; then
        lens_path="$(hf download \
            "$lens_repo_id" \
            "$lens_filename" \
            --repo-type "$lens_repo_type" \
            --revision "$JLENS_LENS_REVISION" \
            --quiet)"
    else
        lens_path="$(hf download \
            "$lens_repo_id" \
            "$lens_filename" \
            --repo-type "$lens_repo_type" \
            --quiet)"
    fi

    if [ -n "$phone_signatures_filename" ]; then
        if [ -n "$phone_signatures_path" ]; then
            die "set either JLENS_PHONE_SIGNATURES_PATH or JLENS_PHONE_SIGNATURES_FILENAME, not both"
        fi
        printf 'Downloading phone-signature artifact from %s/%s...\n' \
            "$lens_repo_id" "$phone_signatures_filename"
        if [ -n "${JLENS_LENS_REVISION:-}" ]; then
            phone_signatures_path="$(hf download \
                "$lens_repo_id" \
                "$phone_signatures_filename" \
                --repo-type "$lens_repo_type" \
                --revision "$JLENS_LENS_REVISION" \
                --quiet)"
        else
            phone_signatures_path="$(hf download \
                "$lens_repo_id" \
                "$phone_signatures_filename" \
                --repo-type "$lens_repo_type" \
                --quiet)"
        fi
    fi
fi

encoder_lens_path="${JLENS_ENCODER_LENS_PATH:-}"
decoder_lens_path="${JLENS_DECODER_LENS_PATH:-}"
if [ -n "$lens_path" ] && \
    { [ -n "$encoder_lens_path" ] || [ -n "$decoder_lens_path" ]; }; then
    die "a combined lens cannot be used with encoder/decoder lens paths"
fi

for artifact in "$lens_path" "$encoder_lens_path" "$decoder_lens_path" \
    "$phone_signatures_path"; do
    if [ -n "$artifact" ] && [ ! -f "$artifact" ]; then
        die "lens artifact does not exist or is not a file: $artifact"
    fi
done

set -- audio-jlens \
    --host 0.0.0.0 \
    --port "$port" \
    --model "${JLENS_MODEL:-openai/whisper-tiny.en}" \
    --device "${JLENS_DEVICE:-auto}"

if [ -n "${JLENS_MODEL_REVISION:-}" ]; then
    set -- "$@" --revision "$JLENS_MODEL_REVISION"
fi
if [ -n "$lens_path" ]; then
    set -- "$@" --lens "$lens_path"
fi
if [ -n "$encoder_lens_path" ]; then
    set -- "$@" --encoder-lens "$encoder_lens_path"
fi
if [ -n "$decoder_lens_path" ]; then
    set -- "$@" --decoder-lens "$decoder_lens_path"
fi
if [ -n "$phone_signatures_path" ]; then
    set -- "$@" --phone-signatures "$phone_signatures_path"
fi
case "${JLENS_ASR_ONLY:-}" in
    ''|0|false|no) ;;
    1|true|yes) set -- "$@" --asr-only ;;
    *) die "JLENS_ASR_ONLY must be true or false" ;;
esac
if [ -n "${JLENS_TOP_K:-}" ]; then
    set -- "$@" --top-k "$JLENS_TOP_K"
fi
if [ -n "${JLENS_TIME_BIN_SECONDS:-}" ]; then
    set -- "$@" --time-bin-seconds "$JLENS_TIME_BIN_SECONDS"
fi
if [ -n "${JLENS_TIME_BIN_OVERLAP_SECONDS:-}" ]; then
    set -- "$@" --time-bin-overlap-seconds "$JLENS_TIME_BIN_OVERLAP_SECONDS"
fi
if [ -n "${JLENS_ANALYSIS_QUEUE_CAPACITY:-}" ]; then
    set -- "$@" --analysis-queue-capacity "$JLENS_ANALYSIS_QUEUE_CAPACITY"
fi
if [ -n "${JLENS_ANALYSIS_QUEUE_INITIAL_SECONDS:-}" ]; then
    set -- "$@" --analysis-queue-initial-seconds "$JLENS_ANALYSIS_QUEUE_INITIAL_SECONDS"
fi
if [ -n "${JLENS_WEB_DIR:-}" ]; then
    set -- "$@" --web-dir "$JLENS_WEB_DIR"
fi
if [ -n "${JLENS_SAMPLES_DIR:-}" ]; then
    set -- "$@" --samples-dir "$JLENS_SAMPLES_DIR"
fi

if [ -n "$lens_path$encoder_lens_path$decoder_lens_path" ]; then
    printf 'Starting Audio Jacobian Lens with model-backed analysis on port %s.\n' \
        "$port"
else
    printf 'Starting Audio Jacobian Lens in demo mode on port %s.\n' "$port"
fi

exec "$@"
