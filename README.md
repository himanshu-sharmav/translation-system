# Machine Translation Backend System

A scalable, cost-efficient, and performant backend system for machine translation services. This system handles translation requests efficiently, scales dynamically based on load, and provides priority handling for urgent translation jobs while maintaining cost optimization during idle periods.

## Features

- **High Performance**: Achieves 1,500+ words per minute translation speed
- **Auto-scaling**: Dynamic resource management based on demand
- **Priority Handling**: Critical, high, and normal priority queues
- **Multi-level Caching**: L1, L2, and L3 caching for optimal performance
- **Cost Optimization**: Intelligent resource pooling and spot instance usage
- **Comprehensive Monitoring**: Prometheus metrics, structured logging, and alerting
- **Security**: JWT authentication, rate limiting, and data encryption
- **RESTful API**: Clean API design with OpenAPI documentation

## Architecture

The system follows a microservices architecture with the following components:

- **API Gateway**: Request routing, authentication, and rate limiting
- **Translation Engine**: GPU/CPU-based translation processing
- **Queue Manager**: Priority-based job queuing and dispatch
- **Cache Manager**: Multi-level caching system
- **Resource Manager**: Auto-scaling and compute instance management
- **Monitoring Service**: Metrics collection and alerting

## Performance Targets

- **Translation Speed**: 1,500 words per minute minimum
- **Large Documents**: 15,000 words processed within 12 minutes
- **Cache Response**: <100ms for cached translations
- **Priority Jobs**: <30 seconds processing time for critical priority

## Cost Estimates

| Daily Volume | Estimated Cost |
|-------------|----------------|
| 10,000 words | $32/month |
| 100,000 words | $316/month |
| 1,000,000 words | $3,477/month |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- PostgreSQL 15+
- Redis 7+

### Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd machine-translation-backend
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the development environment:
```bash
docker-compose up -d
```

4. Run database migrations:
```bash
docker-compose exec api-gateway alembic upgrade head
```

5. The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- API Documentation: `http://localhost:8000/docs`
- Monitoring Dashboard: `http://localhost:3000` (Grafana)
- Metrics: `http://localhost:9091` (Prometheus)

## API Usage

### Submit Translation Request

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source_language": "en",
    "target_language": "es",
    "content": "Hello, world!",
    "priority": "normal"
  }'
```

### Check Job Status

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}" \
  -H "Authorization: Bearer <your-jwt-token>"
```

### Get Translation Result

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/result" \
  -H "Authorization: Bearer <your-jwt-token>"
```

## Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Portuguese (pt)
- Italian (it)
- Russian (ru)

## Configuration

Key configuration options:

```yaml
# Performance
TARGET_WPM: 1500
MAX_DOCUMENT_WORDS: 15000
MAX_PROCESSING_TIME: 12

# Scaling
MIN_INSTANCES: 1
MAX_INSTANCES: 10
SCALE_UP_THRESHOLD: 0.8
SCALE_DOWN_THRESHOLD: 0.2

# Security
JWT_SECRET_KEY: your-secret-key
RATE_LIMIT_PER_MINUTE: 100
```

## Monitoring

The system provides comprehensive monitoring through:

- **Prometheus Metrics**: System performance, queue depths, processing times
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Grafana Dashboards**: Visual monitoring and alerting
- **Cost Tracking**: Real-time cost monitoring and optimization

### Key Metrics

- `translation_requests_per_second`
- `translation_latency_seconds`
- `queue_depth_by_priority`
- `gpu_utilization_percent`
- `cache_hit_ratio`
- `cost_per_translation_usd`

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load tests
pytest tests/load/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint
flake8 src/
mypy src/
```

## Deployment

### Production Deployment

1. Build production images:
```bash
docker build -f Dockerfile.api -t translation-api:latest .
docker build -f Dockerfile.engine -t translation-engine:latest .
```

2. Deploy with Kubernetes:
```bash
kubectl apply -f k8s/
```

3. Configure auto-scaling:
```bash
kubectl apply -f k8s/hpa.yaml
```

### Infrastructure as Code

Terraform templates are provided for AWS deployment:

```bash
cd terraform/
terraform init
terraform plan
terraform apply
```

## Security

- JWT-based authentication
- Rate limiting per user/API key
- Data encryption in transit and at rest
- Audit logging for all operations
- API key management
- Input validation and sanitization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review the troubleshooting guide

## Roadmap

- [ ] Support for additional language pairs
- [ ] Advanced model optimization techniques
- [ ] Real-time translation streaming
- [ ] Multi-region deployment
- [ ] Advanced cost optimization algorithms