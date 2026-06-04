# Infrastructure Runbook: VPC Egress & Cloud NAT

> **Status: DECOMMISSIONED (2026-06-02).** etcher no longer uses Direct VPC
> Egress or the shared Cloud NAT. The whole stack below (`fastfuels-run-subnet`,
> `fastfuels-router`, `fastfuels-nat`, the static NAT IP) was torn down once
> etcher cut over to reading OSM from the GCS FlatGeobuf snapshot
> (`etcher/osm_source.py`) instead of the Overpass API. No Cloud Run service
> currently routes through a VPC subnet. The provisioning steps are retained
> for historical reference / if egress is ever needed again; the teardown that
> was actually run is recorded in the **Decommissioning** section at the bottom.

This runbook documents the manual `gcloud` commands used to provision the static IP routing for our FastFuels Cloud Run backend jobs.

**Prerequisites**
Ensure your Google Cloud CLI is authenticated and configured to the correct project before running these commands:
* `gcloud auth login`
* `gcloud config set project silvx-fastfuels`

**1. Create the VPC Subnet**
We use a `/24` subnet to allow for up to 254 concurrent container instances.

```bash
gcloud compute networks subnets create fastfuels-run-subnet \
  --network=default \
  --region=us-west1 \
  --range=10.10.0.0/24
```


**2. Reserve the Static IP Address**
This is the public-facing IP address that external APIs will see.

```bash
gcloud compute addresses create fastfuels-nat-ip \
  --region=us-west1
```


**3. Create the Cloud Router**
Required to manage the routing logic for the NAT gateway.

```bash
gcloud compute routers create fastfuels-router \
  --network=default \
  --region=us-west1
```

**4. Create the Cloud NAT Gateway**
This attaches the static IP to the router and restricts the NAT explicitly to our Cloud Run subnet.

```bash
gcloud compute routers nats create fastfuels-nat \
  --router=fastfuels-router \
  --region=us-west1 \
  --nat-external-ip-pool=fastfuels-nat-ip \
  --nat-custom-subnet-ip-ranges=fastfuels-run-subnet
```


**5. Update the Cloud Run Jobs**
These commands re-deploy the jobs to use Direct VPC Egress, forcing all external traffic through the new subnet and out the NAT gateway.


# Update Dev
```bash
gcloud run services update etcher-v2-prod \
    --region=us-west1 \
    --network=default \
    --subnet=fastfuels-run-subnet \
    --vpc-egress=all-traffic
```

# To set up queue
```bash
gcloud tasks queues create etcher-v2-queue --location=us-west1
```

# To allow authentication
```bash
 gcloud run services add-iam-policy-binding etcher-v2-prod \
    --region=us-west1 \
    --member="allUsers" \
    --role="roles/run.invoker"
```

## Decommissioning VPC Egress & NAT (post-Overpass migration) — DONE 2026-06-02

The Direct VPC Egress / static-NAT setup above existed **only** to route etcher's
Overpass traffic through a non-blocked IP. etcher no longer calls Overpass — it
reads a static OSM FlatGeobuf snapshot from GCS (see `etcher/osm_source.py`), and
GCS/Firestore are reachable without VPC egress. The cutover landed in prod via PR
#298, and the egress + shared NAT infra were removed on 2026-06-02. The steps
below are what was run, kept for the record.

The CI deploy (`.github/workflows/etcher.yml`) only sets `--update-env-vars`, so it
neither adds nor removes egress — this was a manual action.

**1. Cleared Direct VPC Egress from the etcher service.**

```bash
# Confirm the current networking config first
gcloud run services describe etcher-v2-prod --region=us-west1 \
  --format='yaml(spec.template.metadata.annotations)'

gcloud run services update etcher-v2-prod --region=us-west1 --clear-network
```

This produced network-less revision `etcher-v2-prod-00021-6px`. A post-clear smoke
run (`DEPLOYMENT_ENV=prod uv run pytest tests/integration/handlers/test_osm.py`)
confirmed etcher still reads OSM and writes non-empty road/water parquet without
egress.

**2. Tore down the shared NAT infra — after confirming no other service used it.**

The subnet/router/NAT/IP were shared. We first verified no Cloud Run service in any
region still routed through `fastfuels-run-subnet` (etcher was the last one):

```bash
for s in etcher exporter standgen treevox griddle; do
  echo "== $s-v2-prod =="
  gcloud run services describe "$s-v2-prod" --region=us-west1 \
    --format='value(spec.template.metadata.annotations)' 2>/dev/null \
    | grep -i 'vpc-access\|network\|subnet' || echo "  (no VPC egress)"
done
```

All clear, so the stack was deleted (note the in-use NAT IP was `fastfuels-nat-ip-2`
after an earlier rotation, not `fastfuels-nat-ip`). The idle Direct-VPC-egress
internal address (`serverless-ipv4-*`, 10.10.0.16) auto-released after step 1, so it
did not block the subnet delete:

```bash
gcloud compute routers nats delete fastfuels-nat --router=fastfuels-router --region=us-west1
gcloud compute routers delete fastfuels-router --region=us-west1
gcloud compute addresses delete fastfuels-nat-ip-2 --region=us-west1
gcloud compute networks subnets delete fastfuels-run-subnet --region=us-west1
```
