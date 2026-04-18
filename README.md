##Prerequisites & Setup

Before running the pipeline or deploying this project, you must complete the following preliminary steps in your AWS environment:

### 1. Create an S3 Bucket
You need to manually create an Amazon S3 bucket to store the project's data.
* **Bucket Name:** `agri-artifacts`
*(Note: Please ensure this bucket is created before triggering the GitHub Actions workflow).*

### 2. Update SNS Notification Email
Open the `template.yaml` file located in the repository.
* Locate the **SNS** (Simple Notification Service) section.
* Replace the placeholder email with **your own email address**. This is required to receive pipeline status alerts and notifications.
