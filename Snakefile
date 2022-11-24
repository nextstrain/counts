import os

KNOWN_MODELS = ["mlr", "renewal"]

if not config:
    configfile: "config/config.yaml"

if not config.get("data_provenances"):
    print("ERROR: config must include 'data_provenances'.")
    sys.exit(1)

if not config.get("geo_resolutions"):
    print("ERROR: config must include 'geo_resolutions'.")
    sys.exit(1)

if config.get("send_slack_notifications"):
    # Check that the required environment variables are set for Slack notifications
    required_envvar = ["SLACK_TOKEN", "SLACK_CHANNELS"]
    if any(envvar not in os.environ for envvar in required_envvar):
        print(f"ERROR: Must set the following environment variables to send Slack notifications: {required_envvar}")
        sys.exit(1)

wildcard_constraints:
    date = r"\d{4}-\d{2}-\d{2}",
    geo_resolution = "global|usa"

def get_todays_date():
    from datetime import datetime
    date = datetime.today().strftime('%Y-%m-%d')
    return date

def _get_all_input(w):
    data_provenances = config["data_provenances"]
    geo_resolutions = config["geo_resolutions"]

    all_input = [
        *expand(
            "data/{data_provenance}/{geo_resolution}/prepared_cases.tsv",
            data_provenance=data_provenances,
            geo_resolution=geo_resolutions
        ),
        *expand(
            "data/{data_provenance}/{geo_resolution}/prepared_variants.tsv",
            data_provenance=data_provenances,
            geo_resolution=geo_resolutions
        )
    ]

    if config.get("send_slack_notifications"):
        all_input.append("data/notify_on_new_usa_locations.done")
        all_input.extend(expand(
            "data/{data_provenance}/{geo_resolution}/notify/clade_without_variant.done",
            data_provenance=data_provenances,
            geo_resolution=geo_resolutions
        ))

    if config.get("upload"):
        all_input.extend(expand(
            "data/{data_provenance}/{geo_resolution}_case_counts_upload.done",
            data_provenance=data_provenances,
            geo_resolution=geo_resolutions
        ))
        all_input.extend(expand(
            "data/{data_provenance}/{geo_resolution}/nextstrain_clade_counts_upload.done",
            data_provenance=data_provenances,
            geo_resolution=geo_resolutions
        ))

    # Check which models to run based on which model configs have been provided
    models_to_run = [
        model_name
        for model_name in KNOWN_MODELS
        if config.get(f"{model_name}_config")
    ]

    if models_to_run:
        run_date = config.get("run_date", get_todays_date())
        all_input.extend(expand(
            "results/{data_provenance}/{geo_resolution}/{model}/{date}_results.json",
            data_provenance=data_provenances,
            geo_resolution=geo_resolutions,
            model=models_to_run,
            date=run_date
        ))
        if config.get("upload"):
            all_input.extend(expand(
                "results/{data_provenance}/{geo_resolution}/{model}/{date}_results_s3_upload.done",
                data_provenance=data_provenances,
                geo_resolution=geo_resolutions,
                model=models_to_run,
                date=run_date
            ))

    return all_input


rule all:
    input: _get_all_input


include: "workflow/snakemake_rules/prepare_data.smk"
include: "workflow/snakemake_rules/models.smk"

if config.get("send_slack_notifications"):
    include: "workflow/snakemake_rules/slack_notifications.smk"

if config.get("upload"):
    include: "workflow/snakemake_rules/upload.smk"
