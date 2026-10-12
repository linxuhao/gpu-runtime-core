"""mesa_check: the image's Vulkan userspace against the host's dpkg database."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import mesa_check


def status(*packages):
    return '\n\n'.join('Package: %s\nStatus: %s\nArchitecture: %s\nVersion: %s\nDescription: x\n continued: line'%(name,state,arch,version)
                       for name,version,arch,state in packages)+'\n'


class MesaCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.host,self.own=root/'host-status',root/'own-status'
        self.own.write_text(status(('mesa-vulkan-drivers','25.2.8-0ubuntu0.24.04.4','amd64','install ok installed'),
                                   ('libvulkan1','1.3.275.0-1build1','amd64','install ok installed'),
                                   ('libegl-mesa0','25.2.8-0ubuntu0.24.04.4','amd64','install ok installed')))
    def tearDown(self):self.temp.cleanup()

    def test_only_installed_amd64_versions_count(self):
        text=status(('mesa-vulkan-drivers','25.2.8-0ubuntu0.24.04.2','i386','install ok installed'),
                    ('mesa-vulkan-drivers','25.2.8-0ubuntu0.24.04.4','amd64','install ok installed'),
                    ('mesa-libgallium','25.2.8-0ubuntu0.24.04.1','amd64','deinstall ok config-files'),('python3','3.12','amd64','install ok installed'))
        self.assertEqual(mesa_check.installed(text),{'mesa-vulkan-drivers':'25.2.8-0ubuntu0.24.04.4'})

    def test_a_host_upgrade_is_reported_until_the_image_follows(self):
        check=mesa_check.MesaCheck(str(self.host),str(self.own))
        self.assertEqual(check.warnings(),[],'no host database: nothing to compare')
        self.host.write_text(status(('mesa-vulkan-drivers','25.2.8-0ubuntu0.24.04.4','amd64','install ok installed'),
                                    ('libvulkan1','1.3.275.0-1build1','amd64','install ok installed')))
        self.assertEqual(check.warnings(),[],'libegl-mesa0 is not on the host: not compared')
        self.host.write_text(status(('mesa-vulkan-drivers','25.2.8-0ubuntu0.24.04.5','amd64','install ok installed'),
                                    ('libvulkan1','1.3.275.0-1build1','amd64','install ok installed')))
        os.utime(self.host,ns=(1,10**18))
        [warning]=check.warnings()
        self.assertIn('mesa-vulkan-drivers: host 25.2.8-0ubuntu0.24.04.5, image 25.2.8-0ubuntu0.24.04.4',warning)
        self.assertIn('Mesa packages to the host versions',warning);self.assertNotIn('libvulkan1',warning)


if __name__=='__main__':unittest.main()
