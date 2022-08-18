# Known issue: will translate italic to bold

#open file in read mode
notebook_name="FromSamples2D_banana"
file = open("../python/"+notebook_name+".py", "r")

replaced_content = "%%"+"\n"
start_skipped=0

#looping through the file
for line in file:
    #stripping line break
    line = line.strip()
    if start_skipped<2: #to skip header
        if line == "# ---":
            start_skipped=start_skipped+1
        pass
    else:
        if len(line) > 0:
            if line[0]=="#":
                if line[:4]=="# +":
                    new_line = "%%"
                    replaced_content = replaced_content + new_line + "\n"
                elif line[:4]=="# -":
                    pass
                elif line[:4]=="# # ":
                    new_line = line.replace("# #", "%% ")
                    new_line=new_line.replace("`","|")
                    new_line=new_line.replace("**","*")
                    replaced_content = replaced_content + new_line + "\n"
                elif line[:5]=="# ## ":
                    new_line = line.replace("# ##", "%% ")
                    new_line=new_line.replace("`","|")                    
                    new_line=new_line.replace("**","*")
                    replaced_content = replaced_content + new_line + "\n"                
                elif line[:6]=="# ### ":
                    new_line = line.replace("# ###", "%%")
                    new_line=new_line.replace("`","|")
                    new_line=new_line.replace("**","*")
                    replaced_content = replaced_content + new_line + "\n"
                elif line[:7]=="# #### ":
                    new_line = line.replace("# ####", "%%")
                    new_line=new_line.replace("`","|")
                    new_line=new_line.replace("**","*")
                    replaced_content = replaced_content + new_line + "\n"                
                else:
                    new_line = line.replace("#", "%")
                    new_line=new_line.replace("`","|")
                    new_line=new_line.replace("**","*")
                    replaced_content = replaced_content + new_line + "\n"

    #concatenate the new string and add an end-line break

# Optional line for conversion 
replaced_content = replaced_content + "matlab.internal.liveeditor.openAndConvert('MonotoneLeastSquares.mlx','MonotoneLeastSquares.m')"+"\n"
 
   
#close the file
file.close()
#Open file in write mode
write_file = open(notebook_name+".m", "w")
#overwriting the old file contents with the new/replaced content
write_file.write(replaced_content)
#close the file
write_file.close()