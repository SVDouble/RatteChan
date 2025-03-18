# Tips
### To Convert a Mermaid Diagram to an Image
Run the following command in the terminal:
```bash
mmdc -i figure.mmd -o temp.pdf && pdfcrop temp.pdf figure.pdf && rm temp.pdf
```