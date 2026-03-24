# 05 – CI/CD Pipeline

This module contains the automated CI/CD pipeline for the Apple Segment Classifier.

---

## Overview

Every push to `main` triggers the following pipeline:
```
Push to main → Lint → Tests → Build Docker → Deploy to Render
```

---

## How it works

- **lint-test**: flake8 linting + pytest unit tests
- **build**: builds Docker image (trains the segment classifier inside)
- **deploy**: triggers Render deployment via webhook

---

## Dockerfile

The Dockerfile:
1. Installs all dependencies
2. Trains the segment classifier (`train.py`)
4. Starts the FastAPI server

---

## Live endpoint

https://apple-recommender-v2.onrender.com