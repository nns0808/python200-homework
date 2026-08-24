<!-- Part 1: Warmup — Cloud Concepts -->

<!-- Cloud Concepts Question 1 -->
The core economic model of cloud computing is pay-as-you-go. Instead of purchasing and maintaining your own servers, storage, and networking equipment, you rent computing resources from a cloud provider and pay for what you use.
With your own servers, you must pay upfront for hardware, maintenance, upgrades, electricity, and enough capacity to handle peak demand. With the cloud, you can scale resources up or down as needed and avoid paying for unused hardware.

<!-- Cloud Concepts Question 2 -->
Vertical scaling means making one machine more powerful by adding resources such as CPU, RAM, or a faster GPU. For example, you might choose vertical scaling when a machine learning model needs more RAM or GPU power to finish training faster.
Horizontal scaling means adding more machines and distributing the workload across them. For example, you might choose horizontal scaling when a web application suddenly receives much more traffic and needs multiple servers to handle the increased demand.
Scenarios:
- Web app: Horizontal scaling, because the application can add more machines to handle the increase from 1,000 to 100,000 users.
- Model training: Vertical scaling, because the data scientist needs a more powerful machine with a faster GPU and more RAM.
- Data pipeline: Horizontal scaling, because the 10,000 files can be divided among multiple machines and processed in parallel.


<!-- Cloud Concepts Question 3 -->

Classification:
Gmail: SaaS because it is a complete application that Google manages.
Azure Virtual Machines: IaaS because you rent virtual computing infrastructure and are responsible for configuring and managing the operating system and software.
AWS S3: IaaS because it provides cloud storage infrastructure that you use to store and manage your data.
GitHub Codespaces: PaaS because it provides a managed cloud development environment where you can write and run code without managing the underlying servers.
Snowflake: SaaS because it is a fully managed data platform that users access without managing the underlying infrastructure.
Supabase: BaaS because it provides managed backend services such as databases, authentication, and APIs so developers do not have to build and manage the backend infrastructure themselves.
IaaS, PaaS, and SaaS
IaaS (Infrastructure as a Service) provides basic computing resources such as virtual machines, storage, and networking. For example, Azure Virtual Machines is IaaS. 
PaaS (Platform as a Service) provides a managed environment where I can deploy and run my application without managing the underlying servers. For example, GitHub Codespaces.
SaaS (Software as a Service) is a complete application that I use without managing the infrastructure or application itself. For example, Gmail is SaaS. 

<!-- Cloud Concepts Question 4 -->

A managed data platform like Databricks or Snowflake is a service built to make data engineering, analytics, and machine learning easier by managing much of the underlying cloud infrastructure. Instead of configuring many separate cloud services yourself, you use a platform that provides preconfigured tools for working with data.
The main difference is that AWS or GCP gives you the building blocks, such as compute, storage, databases, and networking, while a managed data platform provides a more integrated environment designed specifically for data workloads.
What you gain:
Easier setup and faster development
Less infrastructure to configure and maintain
Tools specifically designed for data processing, analytics, and ML
Managed scaling and infrastructure
What you give up:
Some flexibility and control over the underlying infrastructure
Potentially higher costs
Greater dependence on the platform and its tools

<!-- Cloud Concepts Question 5 -->

The lesson identifies two situations where the cloud may not be the best choice:
The dataset fits comfortably on a single machine and there are no major compute demands. Local processing may be faster and cheaper.
The project is an initial prototype with relatively small requirements. Using local resources can be simpler and avoid the learning curve and costs of setting up cloud infrastructure


<!-- Part 2: Warmup — Cloud Landscape -->

<!-- Cloud Landscape Question 1 -->

- AWS: AWS offers many cloud services and is widely used by businesses and startups.
- GCP: GCP is strong in data and AI and is often used by companies working with large datasets.
- Azure: Azure works well with Microsoft products and is commonly used by businesses and government organizations.

<!-- Cloud Landscape Question 2 -->

- Access: Supabase is easier to set up because students can create an account quickly without waiting for organizational approval.
- Learning: Supabase uses a SQL database, which helps students learn skills that are useful in many data jobs.
- Pipeline: Supabase makes it eas  to store, check, and connect the raw and processed data in the ETL pipeline.
Reflection: I should choose a cloud tool based on the project's needs, including cost, ease of use, available features, and how well it fits the work.

<!-- Cloud Landscape Question 3 -->

Object storage: AWS S3 can store large amounts of image files and let you access them from different machines.
Compute: AWS EC2 can provide a GPU machine for training and can be shut down when the job is finished.
Serverless compute: Azure Functions can run a web API and automatically adjust to changes in traffic.
LLM API: OpenAI provides an API that can receive structured data and return a text response.

<!-- Cloud Landscape Question 4 -->

My Book Tracker is a simple data project that allows users to add books, store information such as title, author, and reading status, and track their reading progress. The application uses a database to manage the book data and displays it through a web interface.

<!-- Cloud Landscape Question 4 -->

My Book Tracker is a simple data project that allows users to add books, store information such as title, author, and reading status, and track their reading progress. The application uses a database to manage the book data and displays it through a web interface.

A plausible cloud stack could be:

- Supabase for the managed relational database
- AWS S3 for object storage, such as book cover images
- AWS Lambda for serverless compute

Answer:
There is a benefit to consolidating to one provider because it can simplify billing, security, deployment, and management. However, I would give up flexibility because I would be limited to that provider's services instead of choosing the best or most cost-effective service from different providers.