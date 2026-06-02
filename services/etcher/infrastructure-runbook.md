# Infrastructure Runbook: VPC Egress & Cloud NAT

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

## Decommissioning VPC Egress & NAT (post-Overpass migration)

The Direct VPC Egress / static-NAT setup above existed **only** to route etcher's
Overpass traffic through a non-blocked IP. etcher no longer calls Overpass — it
reads a static OSM FlatGeobuf snapshot from GCS (see `etcher/osm_source.py`), and
GCS/Firestore are reachable without VPC egress. Once the FlatGeobuf cutover is
confirmed in prod, the egress and the shared NAT infra can be removed.

The CI deploy (`.github/workflows/etcher.yml`) only sets `--update-env-vars`, so it
neither adds nor removes egress — this is a manual action.

**1. Clear Direct VPC Egress from the etcher service.**

```bash
# Confirm the current networking config first
gcloud run services describe etcher-v2-prod --region=us-west1 \
  --format='yaml(spec.template.metadata.annotations)'

gcloud run services update etcher-v2-prod --region=us-west1 --clear-network
```

Re-run a smoke feature job afterward to confirm etcher still reads OSM and writes
features without egress.

**2. Tear down the shared NAT infra — only after confirming no other service uses it.**

The subnet/router/NAT/IP were shared. Before deleting, verify no sibling service
(`exporter-v2-prod`, `standgen-v2-prod`, `treevox-v2-prod`, `griddle-v2-prod`, …)
still routes through `fastfuels-run-subnet`:

```bash
for s in etcher exporter standgen treevox griddle; do
  echo "== $s-v2-prod =="
  gcloud run services describe "$s-v2-prod" --region=us-west1 \
    --format='value(spec.template.metadata.annotations)' 2>/dev/null \
    | grep -i 'vpc-access\|network\|subnet' || echo "  (no VPC egress)"
done
```

Only if all are clear:

```bash
gcloud compute routers nats delete fastfuels-nat --router=fastfuels-router --region=us-west1
gcloud compute routers delete fastfuels-router --region=us-west1
gcloud compute addresses delete fastfuels-nat-ip --region=us-west1
gcloud compute networks subnets delete fastfuels-run-subnet --region=us-west1
```
