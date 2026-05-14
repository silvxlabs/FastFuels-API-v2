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
gcloud run jobs update etcher-v2 \
  --region=us-west1 \
  --network=default \
  --subnet=fastfuels-run-subnet \
  --vpc-egress=all-traffic
```
