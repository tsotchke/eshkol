# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Eshkol, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. Email: security@eshkol.ai
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: Next release
  - Low: Backlog

## Security Considerations

### Code Execution
Eshkol compiles to native code via LLVM. User programs have full system access equivalent to any native executable. This is by design for performance.

### Memory Safety
- Arena-based allocation prevents most memory leaks
- Bounds checking on array/list access
- Type system prevents many classes of errors

### Input Validation
When processing untrusted input (files, network data), users should:
- Validate input before processing
- Use appropriate error handling
- Consider resource limits

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve Eshkol's security (with their permission).
