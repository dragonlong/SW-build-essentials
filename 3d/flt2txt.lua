-------------------------------------------------------
-- flt2txt.lua                                       --
--                                                   --
-- Remo 3D command-line script for printing nodes    --
-- as an indented scene graph in a new text file.    --
--                                                   --
-- Copyright (c) 2011, Remograph AB.                 --
-- All rights reserved.                              --
--                                                   --
-- This script is provided as free software.         --
-- No responsibility is assumed by Remograph for the --
-- use of this script.                               --
-------------------------------------------------------

local indent = 0
local fileHandle = nil

function preCallback(type)
  -- fileHandle:write(string.rep("  ", indent) .. type .. ": "..remo.getAttributes("Name") .. "\n")
  if (type=="DOF") then
    fileHandle:write(string.rep("  ", indent) .. type .. ": "..remo.getAttributes("Name") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "X: "..remo.getAttributes("ORIGIN X") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "Y: "..remo.getAttributes("ORIGIN Y") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "Z: "..remo.getAttributes("ORIGIN Z") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "Px: "..remo.getAttributes("Point On X Axis X") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "Py: "..remo.getAttributes("Point On X Axis Y") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "Pz: "..remo.getAttributes("Point On X Axis Z") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "XYx: "..remo.getAttributes("Point In XY Plane X") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "XYy: "..remo.getAttributes("Point In XY Plane Y") .. "\n")
    fileHandle:write(string.rep("  ", indent) .. "XYz: "..remo.getAttributes("Point In XY Plane Z") .. "\n")
  end
  indent = indent + 1
  return (type ~= "POLYGON" and type ~= "LIGHTPOINT")
end

function postCallback(type)
  indent = indent - 1
end


------------------
-- main program --
------------------

-- Check the number of command-line arguments and print usage information if incorrect
if (#arg ~= 2) then
  print("Usage:")
  print("flt2txt.lua <model filename> <text filename>")
  print()
  print("<model filename> is the input OpenFlight model file.")
  print("<text filename> is the output text file.")
  print()
  return
end

-- Retrieve arguments
local modelFilename = arg[1]
local textFilename = arg[2]

-- Load model
if (remo.openModel(modelFilename) == false) then
  return
end

-- Open text file for output
fileHandle = io.open(textFilename, "w")
if (fileHandle == nil) then
  print("FAILED opening", textFilename, "for writing.")
  return
end

-- Traverse model, printing node types and names indented
remo.selectAll("DOF")
remo.traverse(preCallback, postCallback)
