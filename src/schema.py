def infer_schema(df):
    schema = dict()
    for c in df.columns:

        if df[c].dtype in [
            "int",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "int8",
            "int16",
            "int32",
            "int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
        ]:
            schema[c] = "NUMERIC"

        elif df[c].dtype in ["float", "Float32", "Float64"]:
            schema[c] = "NUMERIC"

        elif df[c].dtype in ["str", "string", "O", "object"]:
            if df[c].str.len().max() > 100:
                schema[c] = "STRING"
            elif df[c].nunique() / df.shape[0] < 0.1:
                schema[c] = "CATEGORICAL"
            else:
                schema[c] = "STRING"

        elif df[c].dtype == "category":
            schema[c] = "CATEGORICAL"

        elif df[c].dtype == "datetime64":
            schema[c] = "DATETIME"

        elif df[c].dtype == "timedelta64":
            schema[c] = "timedelta"

        elif df[c].dtype == "bool":
            schema[c] = "BINARY"

        else:
            schema[c] = "UNKNOWN"

    return schema
