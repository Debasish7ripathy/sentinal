# Contributing to SENTINEL

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Run `bash setup.sh` to set up the environment
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Workflow

```bash
source venv/bin/activate
python run.py serve --reload    # Auto-reload on code changes
```

## Areas for Contribution

- **ML Models**: Improve threat classification accuracy, add new model architectures
- **IoT Firmware**: Support for additional sensor types or microcontrollers
- **Smart Home**: Integrations with more platforms (Alexa, Google Home, etc.)
- **Frontend**: UI improvements, mobile-responsive design
- **Testing**: Unit tests, integration tests, model evaluation benchmarks
- **Documentation**: Tutorials, wiring guides, deployment guides

## Code Style

- Python: Follow PEP 8
- Arduino/C++: Use consistent formatting matching existing firmware
- Commit messages: Use conventional commits (`feat:`, `fix:`, `docs:`, etc.)

## Pull Requests

1. Keep PRs focused on a single change
2. Update relevant documentation
3. Add tests where applicable
4. Ensure `python run.py status` shows no errors

## Reporting Issues

- Use GitHub Issues
- Include your OS, Python version, and relevant logs
- For IoT issues, include your hardware setup details
