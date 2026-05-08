#!/usr/bin/env bash
# Post-build deploy: bring up container, reload Caddy, wait for cert,
# flip CF proxy=true, purge cache, run health & verification checks.
set -euo pipefail
. /root/seb/.env

cd /root/rxnim

echo "=== bringing up rxnim container ==="
docker compose up -d
sleep 3
docker compose ps

echo
echo "=== waiting for /healthz on localhost:20040 ==="
for i in $(seq 1 60); do
    if curl -fs http://localhost:20040/healthz > /dev/null 2>&1; then
        echo "  /healthz responded after ${i}x2s"
        break
    fi
    sleep 2
done
curl -s http://localhost:20040/healthz || { echo "FAILED to reach /healthz"; exit 1; }
echo

echo "=== /api/health ==="
curl -s http://localhost:20040/api/health | python3 -m json.tool || true

echo
echo "=== reloading Caddy in box-web ==="
cd /root/box/app
docker compose exec -T web caddy reload --config /etc/caddy/Caddyfile
echo "  Caddy reloaded."
cd /root/rxnim

echo
echo "=== waiting for LE cert on rxnim.sebland.com ==="
for i in $(seq 1 30); do
    if curl -ks --max-time 5 https://rxnim.sebland.com/healthz > /dev/null 2>&1; then
        echo "  HTTPS up after ${i}x5s"
        break
    fi
    sleep 5
done

echo
echo "=== flipping CF proxy to true ==="
RECORD_ID="$(curl -fsSL "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records?name=rxnim.sebland.com" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" | python3 -c "import sys,json; print(json.load(sys.stdin)['result'][0]['id'])")"
echo "  record id: ${RECORD_ID}"
curl -fsSL -X PATCH "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records/${RECORD_ID}" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data '{"proxied":true}' | python3 -m json.tool | head -15

echo
echo "=== purging CF cache ==="
curl -fsSL -X POST "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/purge_cache" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data '{"purge_everything":true}' | python3 -m json.tool | head -10

echo
echo "=== final URL check (post-CF-proxy) ==="
sleep 5
/root/seb/scripts/verify-url https://rxnim.sebland.com/ || true
/root/seb/scripts/verify-url https://rxnim.sebland.com/api/health || true

echo
echo "=== non-regression checks ==="
/root/seb/scripts/verify-url https://sebland.com/ || true
/root/seb/scripts/verify-url https://mbta.sebland.com/ || true
/root/seb/scripts/verify-url https://hypno.sebland.com/ || true
/root/seb/scripts/verify-url https://tonykudo.sebland.com/ || true

echo
echo "=== DEPLOY DONE ==="
