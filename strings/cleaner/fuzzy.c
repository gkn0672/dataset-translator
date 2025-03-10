#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <string.h>
#include <stdio.h>

/**
 * Calculates the similarity between two strings.
 *
 * This function compares two strings character by character and counts the number of matches.
 * The similarity is then calculated as the ratio of matches to the length of the strings.
 *
 * @param str1 The first string to compare.
 * @param str2 The second string to compare.
 * @param length The length of the strings to compare.
 * @return The similarity between the two strings as a double value.
 */
static inline double string_similarity(const char *str1, const char *str2, int length)
{
    int matches = 0;
    for (int i = 0; i < length; i++)
    {
        matches += (str1[i] == str2[i]);
    }
    return (double)matches / length;
}

/**
 * Removes fuzzy repeating suffix from the given text and calculates the percentage of the original string that was removed.
 *
 * @param self The Python object representing the current instance.
 * @param args The input arguments passed to the function.
 * @return A tuple containing the cleaned text and the percentage of the original string that was removed.
 */
static PyObject *remove_fuzzy_repeating_suffix(PyObject *self, PyObject *args)
{
    const char *text;
    double threshold;
    Py_ssize_t n;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "s#d", &text, &n, &threshold))
    {
        return NULL;
    }

    if (n == 0 || threshold < 0 || threshold > 1)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input: text cannot be empty, and threshold must be between 0 and 1.");
        return NULL;
    }

    char *cleaned_text = (char *)PyMem_Malloc(n + 1);
    if (!cleaned_text)
    {
        return PyErr_NoMemory();
    }
    memcpy(cleaned_text, text, n);
    cleaned_text[n] = '\0'; // Ensure null-termination

    Py_ssize_t original_length = n;

    for (Py_ssize_t i = 1; i <= n / 2; i++)
    {
        const char *suffix = cleaned_text + n - i;
        const char *remaining_text = cleaned_text + n - 2 * i;

        if (remaining_text < cleaned_text)
        {
            break; // Safety check to ensure no invalid memory access
        }

        double similarity = string_similarity(remaining_text, suffix, i);

        if (similarity >= threshold)
        {
            n -= i;
            cleaned_text[n] = '\0';
            i = 0; // Reset i to check for multiple repeating suffixes
        }
    }

    // Check if the resulting string length `n` matches the UTF-8 encoding expectations
    PyObject *cleaned_result = PyUnicode_DecodeUTF8(cleaned_text, n, "strict");
    PyMem_Free(cleaned_text);

    if (!cleaned_result)
    {
        PyErr_SetString(PyExc_UnicodeDecodeError, "UTF-8 decoding failed, possibly due to an incomplete or corrupted multibyte sequence.");
        return NULL; // If the decoding fails, return NULL with an appropriate error message
    }

    // Calculate the percentage of the original string that was removed
    double percentage_removed = ((original_length - n) * 100.0) / original_length;

    PyObject *percentage_result = PyFloat_FromDouble(percentage_removed);
    if (!percentage_result)
    {
        Py_DECREF(cleaned_result);
        return NULL; // Handle memory allocation failure
    }

    PyObject *result = PyTuple_Pack(2, cleaned_result, percentage_result);
    Py_DECREF(cleaned_result);
    Py_DECREF(percentage_result);

    return result;
}

// Method definition object for this extension
static PyMethodDef FuzzyMethods[] = {
    {"remove_fuzzy_repeating_suffix", remove_fuzzy_repeating_suffix, METH_VARARGS, "Remove fuzzy repeating suffix and return percentage of the original string that was removed"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef fuzzymodule = {
    PyModuleDef_HEAD_INIT,
    "fuzzy",
    "A module to remove fuzzy repeating suffixes and calculate the percentage of the original string that was removed",
    -1,
    FuzzyMethods};

// Module initialization function
PyMODINIT_FUNC PyInit_fuzzy(void)
{
    return PyModule_Create(&fuzzymodule);
}
