#!/bin/bash
# =============================================================================
# Life OS — Automated Database Backup
# Dumps MariaDB from the Docker container and uploads to GCP Storage.
# =============================================================================

# --- Configuration ---
PROJECT_DIR="$HOME/Github/data-playground-base/internal_dataplayground"
BACKUP_DIR="$PROJECT_DIR/db_backups"
GCP_BUCKET="gs://life-os-db-backups"
DB_CONTAINER="life_os_db"
DB_NAME="jobs"
DB_USER="root"
RETAIN_DAYS=7  # How many days of LOCAL backups to keep

# --- Setup ---
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FILENAME="life_os_backup_${TIMESTAMP}.sql.gz"
FILEPATH="$BACKUP_DIR/$FILENAME"

# --- Log helper ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "========================================="
log "Starting Life OS DB backup..."

# --- Step 1: Dump the database from inside the Docker container ---
log "Running mysqldump on container: $DB_CONTAINER..."

# We fetch the root password from the .env file so no plaintext credentials
# are ever inside this script
source "$PROJECT_DIR/.env"

export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_DIR/impactful-post-292301-17bfe2bceb2c.json"
gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet

docker exec "$DB_CONTAINER" \
    mysqldump -u "$DB_USER" -p"${DB_ROOT_PASSWORD}" \
    --single-transaction \
    --routines \
    --triggers \
    "$DB_NAME" | gzip > "$FILEPATH"

# Check the dump actually worked before uploading
if [ $? -ne 0 ] || [ ! -s "$FILEPATH" ]; then
    log "ERROR: mysqldump failed or produced an empty file. Aborting."
    exit 1
fi

FILESIZE=$(du -sh "$FILEPATH" | cut -f1)
log "Dump successful. File: $FILENAME ($FILESIZE)"

# --- Step 2: Upload to GCP ---
log "Uploading to GCP bucket: $GCP_BUCKET..."

gsutil cp "$FILEPATH" "$GCP_BUCKET/$FILENAME"

if [ $? -ne 0 ]; then
    log "ERROR: GCP upload failed. Local backup retained at $FILEPATH"
    exit 1
fi

log "Upload successful."

# --- Step 3: Clean up local backups older than RETAIN_DAYS ---
log "Cleaning up local backups older than $RETAIN_DAYS days..."
find "$BACKUP_DIR" -name "life_os_backup_*.sql.gz" -mtime +$RETAIN_DAYS -delete
log "Cleanup done."

log "Backup complete. ✅"
log "========================================="