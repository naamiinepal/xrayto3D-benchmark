from pathlib import Path
from dotmap import DotMap

# adapted from https://gitlab.com/jeandumoncel/tools-for-deformetrica/-/tree/master/

def write_data_set_xml(root_directory:Path, xml_parameters,file_format='nii.gz'):
    list_sample_files = list(root_directory.glob(f'{ xml_parameters.object_dir}/*.{file_format}'))
    file = open(root_directory / "data_set.xml", "w")
    file.write("<?xml version=\"1.0\"?>\n")
    file.write("<data-set>\n")
    for sample_file, index in zip(list_sample_files, range(len(list_sample_files))):
        file.write("    <subject id=\"%s%d\">\n" % (xml_parameters.subject_id_prefix, index))
        file.write("        <visit id=\"%s\">\n" % xml_parameters.visit_ids)
        file.write("            <filename object_id=\"%s\">%s/%s</filename>\n" % (xml_parameters.object_id,
                    xml_parameters.object_dir, sample_file.name))
        file.write("        </visit>\n")
        file.write("    </subject>\n")
    file.write("</data-set>\n")
    file.close()

def write_model_xml(root_directory, xml_parameters):
    file = open(root_directory / "model.xml", "w")
    file.write("<?xml version=\"1.0\"?>\n")
    file.write("<model>\n")
    file.write("    <model-type>%s</model-type>\n" % xml_parameters.model_type)
    file.write("    <dimension>%s</dimension>\n" % xml_parameters.dimension)
    file.write("    <template>\n")
    file.write("        <object id=\"%s\">\n" % xml_parameters.object_id)
    file.write("            <deformable-object-type>%s</deformable-object-type>\n" %
               xml_parameters.deformable_object_type)
    file.write("            <noise-std>%s</noise-std>\n" % xml_parameters.noise_std)
    file.write("            <filename>%s</filename>\n" % ( xml_parameters.filename))
    file.write("        </object>\n")
    file.write("    </template>\n")
    file.write("    <deformation-parameters>\n")
    file.write("        <kernel-width>%s</kernel-width>\n" % xml_parameters.deformation_kernel_width)
    file.write("        <kernel-type>%s</kernel-type>\n" % xml_parameters.kernel_type)
    file.write("    </deformation-parameters>\n")
    file.write(" </model>\n")
    file.close()


if __name__ == '__main__':
    dataset_parameters = DotMap({'object_dir':'cleaned','subject_id_prefix':'sub','visit_ids':'t0','object_id':'img'})
    root_dir = Path('atlas/femur/data')
    model_parameters = DotMap({'model_type':'DeterministicAtlas','dimension':3,'object_id':'img',
    'deformable_object_type':'Image','object_dir':'.','filename':'mean_template.nii.gz','noise_std':0.1,'kernel_type':'torch','deformation_kernel_width':2})
    write_data_set_xml(root_directory=root_dir,xml_parameters=dataset_parameters,file_format='nii.gz')
    write_model_xml(root_directory=root_dir,xml_parameters=model_parameters)
