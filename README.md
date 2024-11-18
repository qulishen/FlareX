# FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering

### Installation


1. Install dependent packages

    ```bash
    cd FlareX
    pip install -r requirements.txt
    ```

1. Install FlareX<br>
    Please run the following commands in the **FlareX root path** to install FlareX:<br>

    ```bash
    python setup.py develop
    ```
### FlareX structure

```
├── Flare7K
    ├── Flare2D
         ├── input
         ├── gt
    ├── Flare3D
         ├── input
         ├── gt
    ├── test_data
         ├── input
         ├── gt
         ├── mask
```