
## 🔧 Customization

*   **Categories & Keywords:** To add new categories or improve automatic categorization, edit the `CATEGORY_KEYWORDS` dictionary directly within the `app.py` file. You will need to stop and rebuild the Docker container (`docker-compose down && docker-compose up --build`) for changes to take effect.
*   **Excluded Category:** The category excluded from the fortnightly expense calculation (`School Fees` by default) can be changed by modifying the `excluded_category` parameter in the `calculate_fortnightly_expenses` function call within `app.py`.

## 📄 License

*(Optional: Add your license here, e.g., MIT)*

This project is licensed under the MIT License - see the LICENSE file for details (if you add one).