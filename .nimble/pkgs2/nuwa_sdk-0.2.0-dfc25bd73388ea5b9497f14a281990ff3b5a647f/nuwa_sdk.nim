import macros, json, strutils, hashes
import nimpy

proc mapTypeToPython(nimType: string): string =
  ## Maps Nim types to Python type annotations
  case nimType
  of "int", "int64", "int32", "uint", "uint64", "uint32": return "int"
  of "float", "float64", "float32": return "float"
  of "string": return "str"
  of "bool": return "bool"
  of "void": return "None"
  of "auto": return "Any"
  # Handle sequences (list[T])
  elif nimType.startsWith("seq["):
    var inner = nimType[4 .. ^2] # Strip seq[ and ]
    return "list[" & mapTypeToPython(inner) & "]"
  # Handle arrays
  elif nimType.startsWith("array["):
    # array[N, T] - we'll simplify to list[T]
    let parts = nimType[6 .. ^2].split(",")
    if parts.len == 2:
      return "list[" & mapTypeToPython(parts[1].strip()) & "]"
    return "list[Any]"
  else:
    return "Any" # Fallback for unknown types

macro nuwa_export*(prc: untyped): untyped =
  ## Wraps nimpy's exportpy but also generates metadata for stub generation.
  ## If -d:nuwaStubDir=/tmp/path is set, writes JSON files there.
  ## Otherwise, prints "NUWA_STUB:..." to stdout.

  # 1. Validate input is a proc
  prc.expectKind(nnkProcDef)

  let funcName = prc.name.strVal
  var docString = ""
  var args: seq[JsonNode] = @[]
  var returnType = "None"

  # 2. Extract Docstring (collect all consecutive comment statements at start)
  let body = prc.body
  if body.kind == nnkStmtList:
    var docLines = newSeq[string]()
    for stmt in body:
      if stmt.kind == nnkCommentStmt:
        docLines.add(stmt.strVal.strip())
      elif stmt.kind == nnkEmpty:
        continue
      else:
        break # Stop at actual code
    docString = if docLines.len > 0: docLines.join("\n") else: ""

  # 3. Extract Parameters & Return Type
  let params = prc.params
  var firstParam = true

  for param in params:
    if firstParam:
      # First node in params is the Return Type
      if param.kind != nnkEmpty:
        returnType = mapTypeToPython(param.repr)
      firstParam = false
    else:
      # Remaining nodes are arguments (IdentDefs)
      # param structure: [name1, name2, ..., type, defaultVal]
      if param.len >= 2:
        let paramType = param[^2].repr # Type is second to last

        # Check for default value
        let hasDefault = param.len >= 3 and param[^1].kind != nnkEmpty

        let pyType = mapTypeToPython(paramType)

        # Iterate over names (in case of 'proc x(a, b: int)')
        for i in 0 ..< param.len - 2:
          let argName = param[i].strVal
          let argObj = %* {
            "name": argName,
            "type": pyType,
            "hasDefault": hasDefault
          }
          args.add(argObj)

  # 4. Construct JSON Payload
  let payload = %* {
    "name": funcName,
    "args": args,
    "returnType": returnType,
    "doc": docString
  }

  # Convert payload to string once to inject into quote block
  let payloadStr = $payload

  # 5. Generate Compile-Time Logger (The "Hook")
  # We use static: ... to execute this during compilation
  let logger = quote do:
    static:
      # Define the compile-time string flag.
      # Defaults to empty string if not passed by nuwa-build.
      const stubDir {.strdefine: "nuwaStubDir".}: string = ""
      let pStr = `payloadStr`

      if stubDir.len > 0:
        # File-based mode: Write specific JSON file
        # We hash the payload to ensure unique filenames for overloads
        var h = hash(pStr)
        let fname = stubDir & "/" & `funcName` & "_" & $h & ".json"
        writeFile(fname, pStr)
      else:
        # Fallback/Legacy mode: Print to stdout
        echo "NUWA_STUB:" & pStr

  # 6. Return standard nimpy export + our logger
  # Call nimpy's exportpy macro on the proc
  let exportpyCall = newCall(bindSym"exportpy", prc)
  result = quote do:
    `logger`
    `exportpyCall`
